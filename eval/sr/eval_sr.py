"""Compare bicubic and BasicVSR++ checkpoints on the held-out cameras.

    python -m eval.sr.eval_sr --data data/processed/sr_meva
    python -m eval.sr.eval_sr --data data/processed/sr_meva_v2 \
        --models pretrained=models/pretrained/basicvsrpp_reds4.pth v1=models/finetuned/sr_v1/basicvsrpp_meva_v1.pth \
                 v2=models/finetuned/sr_v2/basicvsrpp_meva.pth --tta 8 --report outputs/reports/sr_eval_v2.json

Each model is scored plainly and, with --tta N, also with an N-way flip/transpose self-ensemble ("<name>+tta").
Writes means plus a per-camera breakdown. Defaults reproduce the v1 report (pretrained vs finetuned).
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
    models = dict(m.split("=", 1) for m in a.models) if a.models else {"pretrained": a.pretrained, "finetuned": a.finetuned}
    rep = {}
    for name, w in models.items():
        if not Path(w).exists():
            print(f"skip {name}: {w} not found", flush=True); continue
        model = build(w, dev)
        for tta in ([1, a.tta] if a.tta > 1 else [1]):
            r = evaluate(model, a.data, dev, a.frames, tta=tta)
            rep.setdefault("bicubic", dict(psnr=r["psnr_bicubic"], ssim=r["ssim_bicubic"], n=r["n"],
                                           psnr_by_camera=per_camera(r["per_seq"], "psnr_bicubic")))
            key = name if tta == 1 else f"{name}+tta"
            rep[key] = dict(weights=w, tta=tta, psnr=r["psnr"], ssim=r["ssim"], psnr_by_camera=per_camera(r["per_seq"], "psnr"))
            print(f"{key:14s} psnr {r['psnr']:.3f} ssim {r['ssim']:.4f}  (bicubic {r['psnr_bicubic']:.3f} / {r['ssim_bicubic']:.4f}, n={r['n']})", flush=True)
    base = rep.get("pretrained")
    if base:
        rep["gain_psnr_db"] = {k: round(v["psnr"] - base["psnr"], 4) for k, v in rep.items()
                               if isinstance(v, dict) and "weights" in v and k != "pretrained"}
        for k, g in rep["gain_psnr_db"].items():
            print(f"gain vs pretrained: {k:14s} {g:+.3f} dB", flush=True)
    out = Path(a.report); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/sr_meva")
    ap.add_argument("--models", nargs="*", help="name=path pairs; overrides --pretrained/--finetuned")
    ap.add_argument("--pretrained", default="models/pretrained/basicvsrpp_reds4.pth")
    ap.add_argument("--finetuned", default="models/finetuned/sr/basicvsrpp_meva.pth")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8], help="also score an N-way flip self-ensemble")
    ap.add_argument("--report", default="outputs/reports/sr_eval.json")
    main(ap.parse_args())
