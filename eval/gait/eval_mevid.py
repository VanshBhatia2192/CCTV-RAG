"""MEVID test eval: appearance (zero-shot / fine-tuned CLIP) vs gait (OpenGait) vs fused.
Protocol (ours, MARS-style; NOT the official MEVID script): queries = official query tracklets,
gallery = all other test tracklets; same-ID+same-camera gallery items are junk. 'cc' (cloth-changing)
additionally treats same-ID+same-outfit items as junk. Fusion weight chosen by 2-fold CV over identities."""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path("data/processed/mevid_gait")
A = Path("data/raw/mevid/mevid-v1-annotation-data")
GP = Path("outputs/gait_prep")
GAIT = ["deepgaitv2_grew", "gaitbase_grew", "deepgaitv2_gait3d"]


def embed_stage(a):
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.gait import og
    from src.person import t2p
    man = json.load(open(GP / "test_manifest.json"))
    names = (A / "test_name.txt").read_text().split()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qset = set(man["query_idx"])
    trs = [t for t in man["tracklets"] if t["win"] and (ROOT / "test_sil" / f'{t["i"]}.npz').exists()]
    if a.limit:
        trs = trs[:a.limit]
    print(f"tracklets with silhouettes: {len(trs)} | device {dev}", flush=True)

    def windows(t):
        z = np.load(ROOT / "test_sil" / f'{t["i"]}.npz')
        sil, idx, trunc = z["sil"], z["idx"], z["trunc"]
        ok = sil.reshape(len(sil), -1).any(1)
        if not a.keep_trunc:
            ok = ok & ~trunc
        if a.min_asp:
            ok = ok & (z["asp"].astype(float) >= a.min_asp)
        out = []
        for lo, hi in t["win"]:
            pos = np.searchsorted(idx, np.arange(lo, hi))
            pos = pos[ok[pos]]
            if len(pos) >= a.min_frames:
                out.append(sil[pos])
        return out, float(ok.mean())

    specs = {n: (n, None) for n in a.gait}
    for item in a.gait_ft:
        n, _, ck = item.partition(":"); specs[n + "_ft"] = (n, ck)
    models = {k: og.load(n, dev, ckpt=ck)[0] for k, (n, ck) in specs.items()}
    keep, G, okf = [], {n: [] for n in models}, []
    t0 = time.time()
    for k, t in enumerate(trs):
        ws, frac = windows(t)
        if not ws:
            continue
        keep.append(t); okf.append(frac)
        for n, m in models.items():
            es = []
            for w in ws:
                x = torch.from_numpy(w).float().div(255)[:, :, 10:-10].unsqueeze(0).to(dev)  # [1,s,64,44]
                with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                    es.append(og.embed(m, x)[0].float().cpu())
            G[n].append(torch.stack(es).mean(0).numpy())
        if (k + 1) % 100 == 0:
            print(f"gait {k + 1}/{len(trs)} | {time.time() - t0:.0f}s", flush=True)
    print(f"kept {len(keep)}/{len(trs)} tracklets with >=1 window of >={a.min_frames} clean frames", flush=True)
    del models

    def paths(t):
        idx = np.load(ROOT / "test_sil" / f'{t["i"]}.npz')["idx"]
        sel = idx[np.linspace(0, len(idx) - 1, min(a.n_img, len(idx))).round().astype(int)]
        return [str(ROOT / "test" / t["pid4"] / names[t["s"] + int(j)]) for j in sel]

    pool = ThreadPoolExecutor(8)
    app = {}
    for key, ck in (("app_zs", None), ("app_ft", "models/finetuned/text2person/clip_t2p.pt")):
        m, _ = t2p.load(ck, dev); m.eval(); E = []
        for k, t in enumerate(keep):
            x = torch.stack(list(pool.map(lambda p: t2p.prep(Image.open(p)), paths(t)))).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                f = t2p.img_feat(m, x)
            E.append(F.normalize(F.normalize(f.float(), dim=-1).mean(0), dim=0).cpu().numpy())
            if (k + 1) % 500 == 0:
                print(f"{key} {k + 1}/{len(keep)}", flush=True)
        app[key] = np.stack(E); del m
    arr = dict(tid=[t["i"] for t in keep], pid=[t["pid"] for t in keep], cam=[t["cam"] for t in keep],
               outfit=[t["outfit"] for t in keep], n=[t["n"] for t in keep],
               is_query=[t["i"] in qset for t in keep], ok_frac=okf, **app)
    arr = {k: np.asarray(v) for k, v in arr.items()}
    for n in G:
        arr[f"gait_{n}"] = np.stack(G[n])
    np.savez(a.cache, **arr)
    print("saved", a.cache, flush=True)


def zn(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def gait_sim(eq, eg):
    """OpenGait 'euc': per-part euclidean distance averaged over the 16 parts; returned as -distance."""
    d = 0
    for p in range(eq.shape[2]):
        x, y = eq[:, :, p].astype(np.float64), eg[:, :, p].astype(np.float64)
        d = d + np.sqrt(np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T, 0))
    return -d / eq.shape[2]


def score_stage(a):
    z = np.load(a.cache)
    pid, cam, outfit, n, isq = z["pid"], z["cam"], z["outfit"], z["n"], z["is_query"]
    qi, gi = np.where(isq)[0], np.where(~isq)[0]
    present = [f[5:] for f in z.files if f.startswith("gait_")]
    print(f"eval set: {len(pid)} tracklets ({len(qi)} queries, {len(gi)} gallery), {len(np.unique(pid))} ids | "
          f"mean clean-frame fraction {z['ok_frac'].mean():.2f}")
    S = {k: z[k][qi] @ z[k][gi].T for k in ("app_zs", "app_ft")}
    S["random"] = np.random.RandomState(0).randn(len(qi), len(gi))
    for g in present:
        e = z[f"gait_{g}"]; S[g] = gait_sim(e[qi], e[gi])
        if g.endswith("_ft"):
            f = e.reshape(len(e), -1).astype(np.float64); fq, fg = f[qi], f[gi]
            S[g + "@flatL2"] = -np.sqrt(np.maximum((fq * fq).sum(1)[:, None] + (fg * fg).sum(1)[None] - 2 * fq @ fg.T, 0))
            nq_, ng_ = np.linalg.norm(fq, axis=1, keepdims=True), np.linalg.norm(fg, axis=1, keepdims=True)
            S[g + "@flatCos"] = (fq / nq_) @ (fg / ng_).T

    def ev(Sm, mode):
        res = np.full((len(qi), 4), np.nan)
        for j, q in enumerate(qi):
            same = pid[gi] == pid[q]
            junk = same & (cam[gi] == cam[q])
            if mode == "cc":
                junk = junk | (same & (outfit[gi] == outfit[q]))
            pos = same & ~junk
            if not pos.any():
                continue
            keep = ~junk
            hit = pos[keep][np.argsort(-Sm[j][keep], kind="stable")]
            r = np.where(hit)[0]
            res[j] = [r[0] < 1, r[0] < 5, r[0] < 10, np.mean((np.arange(len(r)) + 1) / (r + 1))]
        return res

    grid = np.linspace(0, 1, 11)
    ids = np.unique(pid[qi]); np.random.RandomState(0).shuffle(ids)
    folds = [ids[:len(ids) // 2], ids[len(ids) // 2:]]

    def cv_fused(Sa, Sg, mode):
        cache = {w: ev(w * zn(Sa) + (1 - w) * zn(Sg), mode) for w in grid}
        res, ws = np.full((len(qi), 4), np.nan), []
        for f in (0, 1):
            tr, te = np.isin(pid[qi], folds[1 - f]), np.isin(pid[qi], folds[f])
            if not (tr & ~np.isnan(cache[1.0][:, 3])).any():
                ws.append(None); continue
            best = max(grid, key=lambda w: np.nanmean(cache[w][tr, 3]))
            res[te] = cache[best][te]; ws.append(round(float(best), 1))
        return res, ws

    pidq = pid[qi]

    def ci(res, B=500):
        v = ~np.isnan(res[:, 3])
        if v.sum() < 5:
            return None
        u = np.unique(pidq[v]); rng = np.random.RandomState(0)
        by = {i: res[v & (pidq == i), 3] for i in u}
        m = [np.concatenate([by[i] for i in rng.choice(u, len(u))]).mean() for _ in range(B)]
        return [round(float(np.percentile(m, 2.5) * 100), 1), round(float(np.percentile(m, 97.5) * 100), 1)]

    def summ(res, mask=None):
        v = ~np.isnan(res[:, 0]) if mask is None else (~np.isnan(res[:, 0]) & mask)
        return dict(n=int(v.sum()), **{k: (round(float(res[v, i].mean() * 100), 1) if v.any() else None)
                                       for i, k in enumerate(["R1", "R5", "R10", "mAP"])})

    qn = n[qi]; b1, b2 = np.percentile(qn, [33.3, 66.6])
    buckets = {f"len<{int(b1)}": qn < b1, f"len {int(b1)}-{int(b2)}": (qn >= b1) & (qn < b2), f"len>={int(b2)}": qn >= b2}
    report = {}
    for mode in ("all", "cc"):
        R, W = {}, {}
        for k in ("random", "app_zs", "app_ft", *present, *[x for x in S if "@flat" in x]):
            R[k] = ev(S[k], mode)
        for g in present:
            R[f"app_ft+{g}"], W[f"app_ft+{g}"] = cv_fused(S["app_ft"], S[g], mode)
        report[mode] = {k: dict(**summ(v), mAP_ci95=ci(v), by_len={b: summ(v, m) for b, m in buckets.items()}) for k, v in R.items()}
        report[mode]["_fusion_weights_appearance(fold0,fold1)"] = W
        title = {"all": "ALL queries (cross-camera)", "cc": "CLOTH-CHANGING (cross-camera + different outfit)"}[mode]
        print(f"\n=== {title} ===")
        print(f"{'method':32s} {'n':>4s} {'R1':>6s} {'R5':>6s} {'R10':>6s} {'mAP':>6s} {'mAP 95% CI (by identity)':>26s} | mAP by query length: " + " / ".join(buckets))
        for k, v in report[mode].items():
            if k.startswith("_"):
                continue
            bl = " / ".join(str(v["by_len"][b]["mAP"]) for b in buckets)
            print(f"{k:32s} {v['n']:4d} {v['R1']!s:>6s} {v['R5']!s:>6s} {v['R10']!s:>6s} {v['mAP']!s:>6s} {str(v['mAP_ci95']):>26s} | {bl}")
        print("fusion weight on appearance (fold0, fold1):", W)
        def pci(d, B=500):
            v = ~np.isnan(d)
            u = np.unique(pidq[v]); rng = np.random.RandomState(0)
            by = {i: d[v & (pidq == i)] for i in u}
            m = [np.concatenate([by[i] for i in rng.choice(u, len(u))]).mean() for _ in range(B)]
            return d[v].mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
        pairs = []
        for k in present:
            if k.endswith("_ft"):
                pairs += [(k, k[:-3]), (k, "app_ft"), (f"app_ft+{k}", "app_ft"), (f"app_ft+{k}", k)]
        pr = {}
        if pairs:
            print("paired differences A - B on the same queries (points; 95% CI, resampling identities):")
            for A_, B_ in pairs:
                if A_ in R and B_ in R:
                    (m3, l3, h3), (m0, l0, h0) = pci(R[A_][:, 3] - R[B_][:, 3]), pci(R[A_][:, 0] - R[B_][:, 0])
                    pr[f"{A_} - {B_}"] = dict(mAP=[round(m3, 1), round(l3, 1), round(h3, 1)], R1=[round(m0, 1), round(l0, 1), round(h0, 1)])
                    print(f"  {A_} - {B_}: mAP {m3:+.1f} [{l3:+.1f}, {h3:+.1f}] | R1 {m0:+.1f} [{l0:+.1f}, {h0:+.1f}]")
        report[mode]["_paired"] = pr
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    out = Path(a.report); out.write_text(json.dumps(report, indent=1)); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["embed", "score", "all"])
    ap.add_argument("--cache", default=str(GP / "emb_test.npz"))
    ap.add_argument("--report", default="outputs/reports/eval_mevid.json")
    ap.add_argument("--gait", nargs="*", default=GAIT)
    ap.add_argument("--gait-ft", nargs="*", default=[], help="model:checkpoint_path, scored as <model>_ft")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--n-img", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-trunc", action="store_true")
    ap.add_argument("--min-asp", type=float, default=0.0)
    a = ap.parse_args()
    if a.stage in ("embed", "all"):
        embed_stage(a)
    if a.stage in ("score", "all"):
        score_stage(a)
