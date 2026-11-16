"""Compare bicubic, the pretrained REDS4 BasicVSR++ and the MEVA fine-tuned one on the held-out cameras.

    python -m eval.sr.eval_sr --data data/processed/sr_meva

Writes outputs/reports/sr_eval.json (means plus per-camera breakdown).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from src.sr.basicvsrpp import build
from training.sr.data import evaluate


def per_camera(rows, k):
    by = defaultdict(list)
    for r in rows:
        by[r["camera"]].append(r[k])
    return {c: round(float(np.mean(v)), 4) for c, v in sorted(by.items())}


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = {}
    for name, w in (("pretrained", a.pretrained), ("finetuned", a.finetuned)):
        if not Path(w).exists():
            print(f"skip {name}: {w} not found", flush=True); continue
        r = evaluate(build(w, dev), a.data, dev, a.frames)
        rep.setdefault("bicubic", dict(psnr=r["psnr_bicubic"], ssim=r["ssim_bicubic"],
                                       psnr_by_camera=per_camera(r["per_seq"], "psnr_bicubic")))
        rep[name] = dict(weights=w, psnr=r["psnr"], ssim=r["ssim"], psnr_by_camera=per_camera(r["per_seq"], "psnr"))
        print(f"{name:10s} psnr {r['psnr']:.3f} ssim {r['ssim']:.4f}  (bicubic {r['psnr_bicubic']:.3f} / {r['ssim_bicubic']:.4f}, n={r['n']})", flush=True)
    if "pretrained" in rep and "finetuned" in rep:
        rep["gain_psnr_db"] = round(rep["finetuned"]["psnr"] - rep["pretrained"]["psnr"], 4)
        print(f"fine-tuning gain: {rep['gain_psnr_db']:+.3f} dB PSNR", flush=True)
    out = Path(a.report); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sr_meva")
    ap.add_argument("--pretrained", default="models/pretrained/basicvsrpp_reds4.pth")
    ap.add_argument("--finetuned", default="models/finetuned/sr/basicvsrpp_meva.pth")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--report", default="outputs/reports/sr_eval.json")
    main(ap.parse_args())
