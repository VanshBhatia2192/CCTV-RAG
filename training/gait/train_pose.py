"""Train the ST-GCN skeleton gait encoder (src/gait/pose_gait.py) on MEVID-train keypoints.

Input: data/processed/mevid_pose/train/*.npz (training/gait/extract_pose.py). Each tracklet window
(64 consecutive frames) is one sequence. Batches are P identities x K sequences, preferring different
outfits/cameras per identity. Loss = label-smoothed cross-entropy on a BNNeck classifier + batch-hard
triplet on the pre-BN feature. A few training identities are held out; validation reports cross-camera
retrieval (all) and the cloth-changing subset (cc, same-outfit matches ignored), and the best
checkpoint by mean(all mAP, cc mAP) is kept.

    python -m training.gait.train_pose --steps 30000
The real test is eval/gait/eval_pose_mevid.py on MEVID test.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.gait.pose_gait import STGCNPoseGait, normalize_skeleton_sequence, flip_keypoints

POSE = Path("data/processed/mevid_pose")
GP = Path("outputs/gait_prep")


def load_split(split, min_len):
    man = json.load(open(GP / f"{split}_manifest.json"))
    data = {}
    for t in man["tracklets"]:
        f = POSE / split / f'{t["i"]}.npz'
        if not t["win"] or not f.exists():
            continue
        z = np.load(f); k, idx, wh = z["kpts"], z["idx"], z["wh"]
        seqs = []
        for lo, hi in t["win"]:
            m = (idx >= lo) & (idx < hi)
            if m.sum() >= min_len:
                seqs.append(dict(k=k[m], w=float(np.median(wh[m, 0])), h=float(np.median(wh[m, 1]))))
        if seqs:
            data.setdefault(t["pid"], []).append(dict(seqs=seqs, outfit=t["outfit"], cam=t["cam"], i=t["i"]))
    return data


def augment(k, w, h, L, rng, a):
    """k (n,17,3) pixel keypoints -> (5, L, 17) features, randomly perturbed."""
    n = len(k)
    speed = rng.uniform(1 / a.speed, a.speed) if a.speed > 1 else 1.0
    span = min(n - 1, (L - 1) * speed)
    s0 = rng.uniform(0, n - 1 - span)
    t = s0 + np.linspace(0, span, L)
    lo = np.floor(t).astype(int); hi = np.minimum(lo + 1, n - 1); f = (t - lo)[:, None, None]
    x = k[lo] * (1 - f) + k[hi] * f
    x[:, :, 2] = np.minimum(k[lo, :, 2], k[hi, :, 2])
    if rng.rand() < 0.5:
        x = flip_keypoints(x, w)
    c = np.array([w / 2, h / 2])
    ang = np.deg2rad(rng.uniform(-a.rot, a.rot)); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    S = np.diag([rng.uniform(1 - a.scale, 1 + a.scale), rng.uniform(1 - a.scale, 1 + a.scale)])
    x[:, :, :2] = (x[:, :, :2] - c) @ (R @ S).T + c
    x[:, :, :2] += rng.randn(L, 17, 2) * a.jitter * h
    drop = rng.rand(L, 17) < a.joint_drop
    drop |= (rng.rand(L) < a.frame_drop)[:, None]
    x[drop, 2] = 0.0
    return normalize_skeleton_sequence(x, a.min_conf)


def pick(tl, K, rng):
    by = {}
    for j, t in enumerate(tl):
        by.setdefault((t["outfit"], t["cam"]), []).append(j)
    for v in by.values():
        rng.shuffle(v)
    keys = list(by); rng.shuffle(keys); out = []
    while len(out) < K and any(by.values()):
        for key in keys:
            if by[key] and len(out) < K:
                out.append(by[key].pop())
    while len(out) < K:
        out.append(int(rng.randint(len(tl))))
    return out


def batch(data, pids, lab, a, rng):
    xs, ys = [], []
    for pid in rng.choice(pids, a.P, replace=False):
        tl = data[pid]
        for j in pick(tl, a.K, rng):
            s = tl[j]["seqs"][rng.randint(len(tl[j]["seqs"]))]
            xs.append(augment(s["k"], s["w"], s["h"], a.L, rng, a)); ys.append(lab[pid])
    return torch.from_numpy(np.stack(xs)).float(), torch.tensor(ys)


def batch_hard_triplet(f, y, margin):
    d = torch.cdist(f, f)
    same = y[:, None] == y[None]
    pos = (d * same.float()).max(1).values
    neg = (d + same.float() * 1e6).min(1).values
    return F.relu(pos - neg + margin).mean(), float((pos < neg).float().mean())


@torch.no_grad()
def embed_tracklets(model, tl, a, dev):
    model.eval(); E = []
    for t in tl:
        es = []
        for s in t["seqs"]:
            x = torch.from_numpy(normalize_skeleton_sequence(s["k"], a.min_conf))[None].to(dev)
            es.append(model(x)[0])
        E.append(F.normalize(torch.stack(es).mean(0), dim=0).cpu().numpy())
    model.train()
    return np.stack(E)


def retrieval(E, pid, cam, outfit, cc=False):
    S = E @ E.T; r1, ap = [], []
    for q in range(len(E)):
        same = pid == pid[q]
        junk = same & (cam == cam[q])
        if cc:
            junk |= same & (outfit == outfit[q])
        junk[q] = True
        pos = same & ~junk
        if not pos.any():
            continue
        keep = ~junk
        hit = pos[keep][np.argsort(-S[q][keep], kind="stable")]
        r = np.where(hit)[0]
        r1.append(r[0] < 1); ap.append(np.mean((np.arange(len(r)) + 1) / (r + 1)))
    return dict(R1=round(float(np.mean(r1)) * 100, 1), mAP=round(float(np.mean(ap)) * 100, 1), n=len(ap)) if ap else dict(R1=None, mAP=None, n=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--P", type=int, default=16); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--L", type=int, default=48, help="training clip length (frames)")
    ap.add_argument("--seq-len", type=int, default=64, help="window length used at inference")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=0.3); ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--speed", type=float, default=1.25); ap.add_argument("--rot", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=0.1); ap.add_argument("--jitter", type=float, default=0.005)
    ap.add_argument("--joint-drop", type=float, default=0.05); ap.add_argument("--frame-drop", type=float, default=0.02)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--val-ids", type=int, default=12); ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/finetuned/gait/stgcn_pose_mevid.pt")
    ap.add_argument("--report", default="outputs/reports/gait_pose_train.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.val_every, a.P, a.K, a.val_ids = 20, 10, 4, 2, 3

    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_split("train", a.min_len)
    ids = sorted(data); np.random.RandomState(0).shuffle(ids)
    val_ids, tr_ids = ids[:a.val_ids], [i for i in ids[a.val_ids:] if len(data[i]) >= 2]
    lab = {p: j for j, p in enumerate(tr_ids)}
    val = [dict(t, pid=p) for p in val_ids for t in data[p]]
    vp, vc, vo = (np.array([t[k] for t in val]) for k in ("pid", "cam", "outfit"))
    print(f"train ids {len(tr_ids)} ({sum(len(data[i]) for i in tr_ids)} tracklets, "
          f"{sum(len(t['seqs']) for i in tr_ids for t in data[i])} windows) | val ids {len(val_ids)} ({len(val)} tracklets) | {dev}", flush=True)

    cfg = dict(out_dim=256, num_classes=len(tr_ids), dropout=a.dropout)
    model = STGCNPoseGait(**cfg).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    def validate():
        E = embed_tracklets(model, val, a, dev)
        return dict(all=retrieval(E, vp, vc, vo), cc=retrieval(E, vp, vc, vo, cc=True))

    def save(path, step, v):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), model_cfg={k: v_ for k, v_ in cfg.items() if k != "num_classes"},
                        seq_len=a.seq_len, min_conf=a.min_conf, step=step, val=v, args=vars(a),
                        keypoints="usyd-community/vitpose-base-simple"), path)

    hist, best, best_score, t0 = [], None, -1.0, time.time()
    v = validate(); print("step 0 (untrained) val:", v, flush=True); hist.append(dict(step=0, **v))
    run = dict(loss=0.0, ce=0.0, tri=0.0, acc=0.0, n=0)
    for step in range(1, a.steps + 1):
        x, y = batch(data, tr_ids, lab, a, rng)
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        _, raw, logits = model(x, return_logits=True)
        ce = F.cross_entropy(logits, y, label_smoothing=a.smooth)
        tri, tacc = batch_hard_triplet(raw, y, a.margin)
        loss = a.ce * ce + tri
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run["loss"] += loss.item(); run["ce"] += ce.item(); run["tri"] += tri.item(); run["acc"] += tacc; run["n"] += 1
        if step % 200 == 0:
            n = run["n"]
            print(f"step {step} | loss {run['loss'] / n:.3f} ce {run['ce'] / n:.3f} tri {run['tri'] / n:.3f} "
                  f"triplet-acc {run['acc'] / n:.2f} | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
            run = dict(loss=0.0, ce=0.0, tri=0.0, acc=0.0, n=0)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(); hist.append(dict(step=step, **v))
            score = np.mean([v["all"]["mAP"] or 0, v["cc"]["mAP"] or 0])
            tag = ""
            if score > best_score:
                best_score, best = score, dict(step=step, **v); save(a.out, step, v); tag = "  <- best, saved"
            print(f"step {step} val: all {v['all']} | cc {v['cc']}{tag}", flush=True)
    save(str(Path(a.out).with_name(Path(a.out).stem + "_last.pt")), a.steps, v)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), best=best, val_history=hist, minutes=round((time.time() - t0) / 60, 1)), open(a.report, "w"), indent=1)
    print("done. best", best, "| saved", a.out, flush=True)


if __name__ == "__main__":
    main()
