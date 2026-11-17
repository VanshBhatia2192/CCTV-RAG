"""Fine-tune BasicVSR++ (from the official REDS4 checkpoint) on MEVA pairs from training/sr/prepare_meva.py.

    python -m training.sr.train --data data/processed/sr_meva --init models/pretrained/basicvsrpp_reds4.pth

Writes models/finetuned/sr/{last.pth,basicvsrpp_meva.pth (best val PSNR)} and outputs/reports/sr_train.json.
Re-running the same command resumes from last.pth (saved every --ckpt-every iterations), so any interruption
costs at most a few minutes of work. `--max-hours` and SIGTERM (job killed / time limit) stop cleanly with a
checkpoint. If the GPU runs out of memory (it is shared with other users) the run saves and exits with code 3,
so the job script can retry with a smaller batch; batch size can change between resumes.
With --ema (default), an exponential moving average of the weights is kept; validation and the saved best
model use the EMA weights, which are smoother and usually score a little higher than the raw weights.
"""
OOM_EXIT = 3
import argparse, copy, json, math, signal, sys, time
from pathlib import Path
import torch
from src.sr.basicvsrpp import BasicVSRPlusPlus, load_weights
from training.sr.data import TrainSet, autocast, evaluate

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}


def charbonnier(a, b, eps=1e-12):
    return torch.sqrt((a - b) ** 2 + eps).mean()


def save(path, model, opt, sched, it, best, hist, ema=None):
    tmp = Path(str(path) + ".tmp")
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), iter=it, best=best, hist=hist,
                    ema=ema.state_dict() if ema is not None else None), tmp)
    tmp.replace(path)


@torch.no_grad()
def ema_update(ema, model, decay, step):
    d = min(decay, (1 + step) / (10 + step))                 # warm up: follow the model closely at first
    ep, mp = list(ema.parameters()), list(model.parameters())
    torch._foreach_mul_(ep, d)
    torch._foreach_add_(ep, [p.detach() for p in mp], alpha=1 - d)
    for eb, mb in zip(ema.buffers(), model.buffers()):
        eb.copy_(mb)


def safe_eval(model, a, dev, amp):
    """Validation, skipped (not fatal) if the shared GPU has no memory for it right now."""
    try:
        return evaluate(model, a.data, dev, a.val_frames, amp)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("  val skipped: GPU out of memory (shared GPU busy)", flush=True)
        return None


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = AMP[a.amp] if dev.type == "cuda" else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep_path = Path(a.report); rep_path.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    model = BasicVSRPlusPlus().to(dev)
    flow = [p for n, p in model.named_parameters() if n.startswith("spynet.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("spynet.")]
    opt = torch.optim.Adam([dict(params=rest, lr=a.lr), dict(params=flow, lr=a.lr * a.flow_lr_mult)], betas=(0.9, 0.99))
    warm = min(500, a.iters // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.iters - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=amp is torch.float16)

    last = out / "last.pth"
    it, best, hist = 0, -1.0, []
    if last.exists() and not a.fresh:
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        it, best, hist = ck["iter"], ck["best"], ck["hist"]
        ema_sd = ck.get("ema")
        print(f"resumed from {last} at iter {it} (best val psnr {best:.3f})", flush=True)
    else:
        assert a.init and Path(a.init).exists(), f"--init checkpoint not found: {a.init}"
        load_weights(model, a.init)
        print(f"init from {a.init}", flush=True)
        v = safe_eval(model, a, dev, amp) if a.val_every else None
        if v:
            hist.append(dict(iter=0, **{k: v[k] for k in v if k != "per_seq"}))
            best = v.get("psnr", -1.0)
            print(f"pretrained val: psnr {v.get('psnr')} ssim {v.get('ssim')} | bicubic psnr {v.get('psnr_bicubic')}", flush=True)
        model.to(dev)
        ema_sd = None
    ema = None
    if a.ema:
        ema = copy.deepcopy(model).eval()
        if ema_sd:
            ema.load_state_dict(ema_sd)
        for p in ema.parameters():
            p.requires_grad_(False)
    val_model = ema if ema is not None else model

    gb = torch.cuda.mem_get_info()[0] / 1e9 if dev.type == "cuda" else 0
    print(f"device {dev} free {gb:.1f} GB | batch {a.bs} x {a.frames} frames x {a.patch}px | amp {a.amp if amp else 'none'} | {a.iters} iters", flush=True)
    dl = torch.utils.data.DataLoader(TrainSet(a.data, a.frames, a.patch), batch_size=a.bs, num_workers=a.workers,
                                     pin_memory=dev.type == "cuda", drop_last=True,
                                     **(dict(persistent_workers=True, prefetch_factor=4) if a.workers else {}))
    t_start, t_log, loss_sum = time.time(), time.time(), 0.0
    stop = False
    term = []
    signal.signal(signal.SIGTERM, lambda *_: term.append(1))   # job killed / walltime: checkpoint and leave
    model.train()
    for lq, gt in dl:
        if it >= a.iters or stop:
            break
        frozen = it < a.fix_flow_iters                    # keep the pretrained flow net fixed early on
        for p in flow:
            p.requires_grad_(not frozen)
        lq, gt = lq.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
        try:
            with autocast(dev, amp):
                loss = charbonnier(model(lq).float(), gt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            del lq, gt
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            save(last, model, opt, sched, it, best, hist, ema)
            print(f"GPU out of memory at iter {it} (batch {a.bs}); saved {last}, exiting with code {OOM_EXIT}", flush=True)
            sys.exit(OOM_EXIT)
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None:
            ema_update(ema, model, a.ema_decay, it)
        it += 1; loss_sum += loss.item()
        if it % a.log_every == 0:
            dt = (time.time() - t_log) / a.log_every; t_log = time.time()
            eta = (a.iters - it) * dt / 3600
            print(f"iter {it}/{a.iters} loss {loss_sum / a.log_every:.5f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{dt:.3f}s/it eta {eta:.1f}h" + (" [flow frozen]" if frozen else ""), flush=True)
            loss_sum = 0.0
        hours = (time.time() - t_start) / 3600
        stop = bool(term) or bool(a.max_hours and hours >= a.max_hours)
        v = None
        if a.val_every and (it % a.val_every == 0 or it == a.iters) and not term:
            v = safe_eval(val_model, a, dev, amp); model.train()
        if v:
            hist.append(dict(iter=it, **{k: v[k] for k in v if k != "per_seq"}))
            print(f"val @ {it}: psnr {v.get('psnr')} ssim {v.get('ssim')} (best {best:.3f})", flush=True)
            if v.get("psnr", -1) > best:
                best = v["psnr"]
                torch.save(dict(model=val_model.state_dict(), iter=it, val=hist[-1], ema=ema is not None), out / "basicvsrpp_meva.pth")
                print(f"  new best -> {out / 'basicvsrpp_meva.pth'}", flush=True)
            json.dump(dict(args=vars(a), best_psnr=best, hist=hist), open(rep_path, "w"), indent=1)
        if it % a.ckpt_every == 0 or it == a.iters or stop:
            save(last, model, opt, sched, it, best, hist, ema)
    if stop:
        why = "job was stopped (SIGTERM)" if term else f"time limit {a.max_hours}h"
        print(f"stopped at iter {it}: {why}; checkpoint saved, re-run the same command to resume", flush=True)
    else:
        print(f"done: {it} iters, best val psnr {best:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sr_meva")
    ap.add_argument("--init", default="models/pretrained/basicvsrpp_reds4.pth", help="official REDS4 checkpoint")
    ap.add_argument("--out", default="models/finetuned/sr")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--patch", type=int, default=64, help="lq crop size; gt crop is 4x")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--flow-lr-mult", type=float, default=0.25)
    ap.add_argument("--fix-flow-iters", type=int, default=5000)
    ap.add_argument("--amp", choices=list(AMP), default="bf16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=5000, help="0 disables validation")
    ap.add_argument("--val-frames", type=int, default=30)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this many hours (0 = no limit)")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pth and start over from --init")
    ap.add_argument("--report", default="outputs/reports/sr_train.json")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="keep an EMA of the weights")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    main(ap.parse_args())
