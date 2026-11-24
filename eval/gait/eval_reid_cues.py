"""Do the multi-cue re-ID ideas from skeleton-gait (6170b1a) help? Measured on MEVID with identity ground truth.

That commit adds to appearance + gait:
  * kinematic gating: cadence (steps/min from skeleton keypoints) and body aspect ratio (box h/w), with a soft
    penalty (cadence diff > 18, aspect diff > 0.5) and, in cloth-change mode, hard rejection (> 35 / > 1.2);
  * spatio-temporal camera topology gating;
  * new fusion weights: 'any' 0.75*app + 0.25*gait + 0.10*kin; cloth-change 0.5*gait + 0.2*app + 0.15*kin.
MEVID has no timestamps, so the topology cue cannot be tested here (and the topology file only lists CityFlow cameras).
Everything else is scored with eval/gait/eval_mevid.py's protocol on the same 1636 tracklets, using main's gait
(DeepGaitV2 fine-tuned, euclidean -> higher-is-better) and appearance (fine-tuned CLIP).

    python -m eval.gait.eval_reid_cues [--cache outputs/gait_prep/emb_test_ft.npz]
"""
import argparse, json
from pathlib import Path
import numpy as np
from eval.gait.eval_mevid import gait_sim, zn

POSE = Path("data/processed/mevid_pose/test")


def cadence(kpts, fps=30.0):
    """Steps/min from the ankle-separation spectrum (same as the branch's compute_kinematic_gait_metrics)."""
    k = np.asarray(kpts, dtype=np.float32)
    if len(k) < 16:
        return None
    stride = np.linalg.norm(k[:, 15, :2] - k[:, 16, :2], axis=1)
    det = stride - stride.mean()
    f = np.abs(np.fft.rfft(det)); fr = np.fft.rfftfreq(len(k), d=1.0 / fps)
    band = (fr >= 1.0) & (fr <= 3.5)
    return float(fr[band][np.argmax(f[band])] * 60.0) if band.any() else None


def cues(tid):
    """(cadence, aspect) for a tracklet: cadence = median over its consecutive-frame windows, aspect = median h/w."""
    p = POSE / f"{int(tid)}.npz"
    if not p.exists():
        return None, None
    z = np.load(p)
    kp, idx, wh = z["kpts"], z["idx"], z["wh"]
    asp = float(np.median(wh[:, 1] / np.maximum(wh[:, 0], 1))) if len(wh) else None
    cuts = np.where(np.diff(idx) != 1)[0] + 1
    cads = [c for seg in np.split(np.arange(len(idx)), cuts) if (c := cadence(kp[seg])) is not None]
    return (float(np.median(cads)) if cads else None), asp


def kin_matrix(cq, cg, aq, ag, strict):
    """The branch's validate_kinematics as a matrix: score in [0.2, 1]; NaN = rejected (strict cloth-change mode)."""
    pen = np.zeros((len(cq), len(cg)))
    rej = np.zeros_like(pen, dtype=bool)
    ok_c = ~np.isnan(cq)[:, None] & ~np.isnan(cg)[None]
    dc = np.abs(cq[:, None] - cg[None])
    pen += np.where(ok_c & (dc > 18), np.minimum(0.35, (dc - 18) / 40), 0)
    ok_a = ~np.isnan(aq)[:, None] & ~np.isnan(ag)[None]
    da = np.abs(aq[:, None] - ag[None])
    pen += np.where(ok_a & (da > 0.5), np.minimum(0.30, (da - 0.5) / 1.5), 0)
    if strict:
        rej |= (ok_c & (dc > 35)) | (ok_a & (da > 1.2))
    kin = np.maximum(0.2, 1 - pen)
    kin[rej] = np.nan
    return kin


def main(a):
    z = np.load(a.cache)
    pid, cam, outfit, isq, tid = z["pid"], z["cam"], z["outfit"], z["is_query"], z["tid"]
    qi, gi = np.where(isq)[0], np.where(~isq)[0]
    g = f"gait_{a.gait}"
    Sa = zn(z["app_ft"][qi] @ z["app_ft"][gi].T)
    Sg = zn(gait_sim(z[g][qi], z[g][gi]))
    cad, asp = map(np.array, zip(*[cues(t) for t in tid]))
    cad = np.array([np.nan if c is None else c for c in cad], float)
    asp = np.array([np.nan if c is None else c for c in asp], float)
    print(f"{len(qi)} queries, {len(gi)} gallery | cadence for {np.mean(~np.isnan(cad)):.0%}, aspect for {np.mean(~np.isnan(asp)):.0%}")
    # how identity-specific are the cues? same-person vs different-person differences
    same = pid[:, None] == pid[None]
    iu = np.triu_indices(len(pid), 1)
    for name, v in (("cadence", cad), ("aspect", asp)):
        d = np.abs(v[:, None] - v[None])[iu]; s = same[iu]; ok = ~np.isnan(d)
        print(f"  |{name} diff| median: same person {np.median(d[ok & s]):.2f}, different {np.median(d[ok & ~s]):.2f}")

    kin_soft = kin_matrix(cad[qi], cad[gi], asp[qi], asp[gi], strict=False)
    kin_strict = kin_matrix(cad[qi], cad[gi], asp[qi], asp[gi], strict=True)
    K = (np.nan_to_num(kin_soft, nan=0.2) - 0.5) * 4
    Ks = (kin_strict - 0.5) * 4                      # NaN where rejected

    V = {
        "main any (0.6 app + 0.4 gait)": 0.6 * Sa + 0.4 * Sg,
        "main cloth-change (gait only)": Sg,
        "branch any (0.75 app + 0.25 gait + 0.1 kin)": 0.75 * Sa + 0.25 * Sg + 0.10 * K,
        "branch any, without kin": 0.75 * Sa + 0.25 * Sg,
        "main any + 0.1 kin": 0.6 * Sa + 0.4 * Sg + 0.10 * K,
        "branch cloth-change (0.5 gait + 0.2 app + 0.15 kin, strict reject)": 0.5 * Sg + 0.2 * Sa + 0.15 * Ks,
        "branch cloth-change, soft kin (no reject)": 0.5 * Sg + 0.2 * Sa + 0.15 * K,
        "branch cloth-change, no kin": 0.5 * Sg + 0.2 * Sa,
        "kin only": K,
        "random": np.random.RandomState(0).randn(len(qi), len(gi)),
    }

    def ev(S, mode):
        res = np.full((len(qi), 4), np.nan)
        for j, q in enumerate(qi):
            same_ = pid[gi] == pid[q]
            junk = same_ & (cam[gi] == cam[q])
            if mode == "cc":
                junk |= same_ & (outfit[gi] == outfit[q])
            pos = same_ & ~junk
            if not pos.any():
                continue
            s = np.where(np.isnan(S[j]), -np.inf, S[j])      # rejected candidates go to the bottom
            keep = ~junk
            hit = pos[keep][np.argsort(-s[keep], kind="stable")]
            r = np.where(hit)[0]
            res[j] = [r[0] < 1, r[0] < 5, r[0] < 10, np.mean((np.arange(len(r)) + 1) / (r + 1))]
        return res

    pidq = pid[qi]

    def ci(d, B=500):
        v = ~np.isnan(d); u = np.unique(pidq[v]); rng = np.random.RandomState(0)
        by = {i: d[v & (pidq == i)] for i in u}
        m = [np.concatenate([by[i] for i in rng.choice(u, len(u))]).mean() for _ in range(B)]
        return [round(float(d[v].mean() * 100), 1), round(float(np.percentile(m, 2.5) * 100), 1), round(float(np.percentile(m, 97.5) * 100), 1)]

    report = {}
    for mode in ("all", "cc"):
        R = {k: ev(S, mode) for k, S in V.items()}
        base = "main any (0.6 app + 0.4 gait)" if mode == "all" else "main cloth-change (gait only)"
        print(f"\n=== {'ALL (cross-camera)' if mode == 'all' else 'CLOTH-CHANGING'} ===   paired difference vs {base!r}")
        print(f"{'method':66s} {'R1':>5s} {'mAP':>5s}   d-mAP [95% CI]")
        report[mode] = {}
        for k, r in R.items():
            v = ~np.isnan(r[:, 0])
            m1, mp = r[v, 0].mean() * 100, r[v, 3].mean() * 100
            d = ci(r[:, 3] - R[base][:, 3])
            print(f"{k:66s} {m1:5.1f} {mp:5.1f}   {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
            report[mode][k] = dict(R1=round(m1, 1), mAP=round(mp, 1), d_mAP_vs_main=d, n=int(v.sum()))
    rej = np.isnan(kin_strict)
    tp = (pid[qi][:, None] == pid[gi][None]) & (outfit[qi][:, None] != outfit[gi][None])
    print(f"\nstrict cloth-change rejection removes {rej.mean():.1%} of all pairs and {rej[tp].mean():.1%} of TRUE "
          f"cloth-changing matches")
    report["strict_reject"] = dict(all_pairs=round(float(rej.mean()), 3), true_cc_matches=round(float(rej[tp].mean()), 3))
    Path(a.report).write_text(json.dumps(report, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/gait_prep/emb_test_ft.npz")
    ap.add_argument("--gait", default="deepgaitv2_gait3d_ft")
    ap.add_argument("--report", default="outputs/reports/eval_reid_cues.json")
    main(ap.parse_args())
