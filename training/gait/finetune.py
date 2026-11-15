"""Fine-tune an OpenGait model on MEVID-train silhouettes: batch-all triplet loss (per part, margin 0.2),
identity-balanced sampling preferring different outfits. Held-out train identities give a validation mAP
(step 0 = zero-shot); the real test is eval/gait/eval_mevid.py on MEVID test."""
import argparse, json, math, time
from pathlib import Path
import numpy as np


def load_split(root, gp, split, min_frames, keep_trunc, min_asp):
    man = json.load(open(Path(gp) / f"{split}_manifest.json"))
    data = {}
    for t in man["tracklets"]:
        f = Path(root) / f"{split}_sil" / f'{t["i"]}.npz'
        if not t["win"] or not f.exists():
            continue
        z = np.load(f); sil = z["sil"]
        ok = sil.reshape(len(sil), -1).any(1)
        if not keep_trunc:
            ok &= ~z["trunc"]
        if min_asp:
            ok &= z["asp"].astype(float) >= min_asp
        if ok.sum() >= min_frames:
            data.setdefault(t["pid"], []).append(dict(sil=sil[ok], outfit=t["outfit"], cam=t["cam"], i=t["i"]))
    return data


def pick_tracklets(tl, K, rng):
    by = {}
    for j, t in enumerate(tl):
        by.setdefault(t["outfit"], []).append(j)
    for v in by.values():
        rng.shuffle(v)
    keys = list(by); rng.shuffle(keys); out = []
    while len(out) < K and any(by.values()):
        for k in keys:
            if by[k] and len(out) < K:
                out.append(by[k].pop())
    while len(out) < K:
        out.append(int(rng.randint(len(tl))))
    return out


def sample_batch(data, pids, P, K, L, rng, flip_p):
    xs, ys = [], []
    for pid in rng.choice(pids, P, replace=False):
        tl = data[pid]
        for j in pick_tracklets(tl, K, rng):
            s = tl[j]["sil"]; n = len(s)
            seq = s[rng.randint(0, n - L + 1):][:L] if n >= L else s[np.arange(L) % n]
            if rng.rand() < flip_p:
                seq = seq[:, :, ::-1]
            xs.append(seq); ys.append(pid)
    return np.stack(xs), np.array(ys)


def triplet_loss(emb, labels, margin=0.2):
    """emb [B,C,P]; batch-all triplets per part, mean over non-zero triplets (OpenGait-style)."""
    import torch, torch.nn.functional as F
    e = emb.permute(2, 0, 1).float()
    sq = (e * e).sum(-1)
    d = torch.sqrt((sq[:, :, None] + sq[:, None, :] - 2 * e @ e.transpose(1, 2)).clamp_min(1e-8))
    same = labels[:, None] == labels[None, :]
    pos, neg = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device), ~same
    l = F.relu(d[:, :, :, None] - d[:, :, None, :] + margin) * (pos[:, :, None] & neg[:, None, :])
    nz = (l > 0).sum((1, 2, 3)).clamp_min(1)
    return (l.sum((1, 2, 3)) / nz).mean(), float((l > 0).float().sum() / (pos[:, :, None] & neg[:, None, :]).sum() / l.shape[0])


def retrieval(E, pid, cam):
    from eval.gait.eval_mevid import gait_sim
    S = gait_sim(E, E); r1, ap = [], []
    for q in range(len(E)):
        same = pid == pid[q]; junk = same & (cam == cam[q]); pos = same & ~junk
        if not pos.any():
            continue
        keep = ~junk
        hit = pos[keep][np.argsort(-S[q][keep], kind="stable")]
        r = np.where(hit)[0]
        r1.append(r[0] < 1); ap.append(np.mean((np.arange(len(r)) + 1) / (r + 1)))
    return dict(R1=round(float(np.mean(r1)) * 100, 1), mAP=round(float(np.mean(ap)) * 100, 1), n=len(ap))


def main():
    import torch
    from src.gait import og
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepgaitv2_gait3d", choices=list(og.MODELS))
    ap.add_argument("--root", default="data/processed/mevid_gait"); ap.add_argument("--gp", default="outputs/gait_prep")
    ap.add_argument("--out", default="outputs/gait_ft"); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--P", type=int, default=12); ap.add_argument("--K", type=int, default=4); ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--flip", type=float, default=0.2)
    ap.add_argument("--val-ids", type=int, default=12); ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--min-frames", type=int, default=20); ap.add_argument("--keep-trunc", action="store_true")
    ap.add_argument("--min-asp", type=float, default=0.0); ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.P, a.K, a.L, a.val_ids, a.val_every = 3, 3, 2, 12, 3, 2
    dev = "cpu" if a.smoke or not torch.cuda.is_available() else "cuda"
    rng = np.random.RandomState(a.seed); torch.manual_seed(a.seed)
    data = load_split(a.root, a.gp, "train", a.min_frames, a.keep_trunc, a.min_asp)
    ids = np.array(sorted(data)); rng.shuffle(ids)
    val_ids, tr_ids = ids[:a.val_ids], ids[a.val_ids:]
    tr_ids = np.array([i for i in tr_ids if len(data[i]) >= 2]) if not a.smoke else tr_ids
    val = [dict(t, pid=p) for p in val_ids for t in data[p]]
    vp, vc = np.array([t["pid"] for t in val]), np.array([t["cam"] for t in val])
    print(f"train ids {len(tr_ids)} ({sum(len(data[i]) for i in tr_ids)} tracklets) | val ids {len(val_ids)} ({len(val)} tracklets) | device {dev}", flush=True)
    assert len(tr_ids) >= a.P, "fewer training identities than --P"

    model, _ = og.load(a.model, dev, pretrained=not a.scratch)
    for p in getattr(model, "BNNecks", torch.nn.Module()).parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=5e-4)
    warm = min(100, a.steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    def inp(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev).float().div_(255)[:, :, :, 10:-10]

    def fwd(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            return model(([x], None, None, None, None))["inference_feat"]["embeddings"]

    @torch.no_grad()
    def validate():
        model.eval(); E = np.stack([fwd(inp(t["sil"][None]))[0].float().cpu().numpy() for t in val]); model.train()
        return retrieval(E, vp, vc)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    best = validate(); print("step 0 (zero-shot) val:", best, flush=True)
    model.train(); t0 = time.time(); hist = [dict(step=0, **best)]; best_map = best["mAP"]
    for step in range(1, a.steps + 1):
        x, y = sample_batch(data, tr_ids, a.P, a.K, a.L, rng, a.flip)
        loss, frac = triplet_loss(fwd(inp(x)), torch.from_numpy(y).to(dev))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.4f} active-triplets {frac:.2f} lr {sched.get_last_lr()[0]:.4f} | {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(); hist.append(dict(step=step, **v)); print(f"step {step} val: {v}", flush=True)
            if v["mAP"] >= best_map:
                best_map = v["mAP"]; torch.save({"model": model.state_dict()}, f"{a.out}/{a.model}_ft.pt")
    if not (Path(a.out) / f"{a.model}_ft.pt").exists():
        torch.save({"model": model.state_dict()}, f"{a.out}/{a.model}_ft.pt")
    json.dump(dict(args=vars(a), val_history=hist, best_val_mAP=best_map), open(f"{a.out}/{a.model}_ft.json", "w"), indent=1)
    print("done. best val mAP", best_map, "| saved", f"{a.out}/{a.model}_ft.pt", flush=True)


if __name__ == "__main__":
    main()
