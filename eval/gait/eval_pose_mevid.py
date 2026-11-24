"""MEVID test eval of the skeleton (ST-GCN) gait encoder, on exactly the tracklets and protocol of the
silhouette eval (eval/gait/eval_mevid.py), so its numbers sit next to DeepGaitV2 and appearance.

Takes the existing embedding cache (outputs/gait_prep/emb_test_ft.npz: app_zs, app_ft, DeepGaitV2 zero-shot
and fine-tuned), adds `gait_stgcn_pose_ft` (embedding per 64-frame window from
data/processed/mevid_pose/test, averaged), and runs the same scoring (all / cloth-changing, appearance
fusion weight by 2-fold CV over identities).

    python -m eval.gait.eval_pose_mevid
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from eval.gait import eval_mevid as EM
from src.gait import pose_gait as pg

POSE = Path("data/processed/mevid_pose/test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", default="outputs/gait_prep/emb_test_ft.npz")
    ap.add_argument("--cache", default="outputs/gait_prep/emb_test_pose.npz")
    ap.add_argument("--report", default="outputs/reports/eval_mevid_pose.json")
    ap.add_argument("--weights", default=pg.DEFAULT_WEIGHTS)
    ap.add_argument("--name", default="stgcn_pose_ft")
    ap.add_argument("--min-frames", type=int, default=20)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = pg.load_encoder(dev, a.weights)
    assert model is not None, f"no weights at {a.weights}"
    man = {t["i"]: t for t in json.load(open(EM.GP / "test_manifest.json"))["tracklets"]}
    z = dict(np.load(a.base_cache))
    E, missing = [], 0
    rnd = np.random.RandomState(0)
    for tid in z["tid"]:
        t, f = man[int(tid)], POSE / f"{int(tid)}.npz"
        es = []
        if f.exists():
            p = np.load(f)
            for lo, hi in t["win"]:
                m = (p["idx"] >= lo) & (p["idx"] < hi)
                if m.sum() >= a.min_frames:
                    e = pg.embed_keypoints(p["kpts"][m], model=model)
                    if e is not None:
                        es.append(e)
        if es:
            v = np.mean(es, 0); E.append(v / np.linalg.norm(v))
        else:      # no usable skeleton: an uninformative random direction
            missing += 1; v = rnd.randn(256); E.append(v / np.linalg.norm(v))
    print(f"skeleton embeddings for {len(E) - missing}/{len(E)} tracklets ({missing} without usable pose)", flush=True)
    z[f"gait_{a.name}"] = np.stack(E).astype(np.float32)[:, :, None]    # [N, 256, 1]: per-part L2 == plain L2
    np.savez(a.cache, **z)
    EM.score_stage(argparse.Namespace(cache=a.cache, report=a.report))
    r = json.loads(Path(a.report).read_text()); r["_meta"] = dict(weights=a.weights, missing_pose=missing, n=len(E))
    Path(a.report).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
