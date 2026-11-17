"""Presentation figures for the BasicVSR++ MEVA fine-tuning run.

    python -m eval.sr.plots                      # after training + eval.sr.eval_sr
    python -m eval.sr.plots --no-visual          # charts only (no model inference)

Reads outputs/logs/sr_train.log (loss every 100 iters), outputs/reports/sr_train.json (val PSNR/SSIM every
5k iters) and, if present, outputs/reports/sr_eval.json (final per-camera numbers). With --data and the two
checkpoints it also renders side-by-side crops (input / bicubic / pretrained / fine-tuned / ground truth).
Writes PNGs (200 dpi, slide-ready) to outputs/figures/.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference palette (light): categorical slots 1-2, neutral reference ink, text and surface tokens
FT, PRE, REF = "#2a78d6", "#eb6834", "#8a8984"
TEXT, TEXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 12,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "legend.labelcolor": TEXT2,
})


def parse_log(path):
    it, loss, lr = [], [], []
    seen = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"iter (\d+)/\d+ loss ([\d.e+-]+) lr ([\d.e+-]+)", line)
        if m:
            seen[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))   # a resumed run overwrites repeats
    for k in sorted(seen):
        it.append(k); loss.append(seen[k][0]); lr.append(seen[k][1])
    return np.array(it), np.array(loss), np.array(lr)


def smooth(y, k=10):
    """Trailing moving average; the first points average over what exists so far."""
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0.0))
    n = np.arange(1, len(y) + 1)
    lo = np.maximum(n - k, 0)
    return (c[n] - c[lo]) / (n - lo)


def kfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))


def unfreeze_marker(ax, at, y, bottom=False):
    ax.axvline(at, color=REF, lw=1, ls=":", zorder=1)
    ax.annotate("flow network unfrozen", (at, y), xytext=(6, 4 if bottom else 0), textcoords="offset points", color=TEXT2,
                fontsize=10, va="bottom" if bottom else "top")


def fig_loss(it, loss, out, unfreeze):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(it, loss, color=FT, lw=1, alpha=0.25)
    ax.plot(it, smooth(loss), color=FT, lw=2)
    ax.set_title("Training loss (Charbonnier) on MEVA")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss, lower is better"); kfmt(ax)
    ax.text(0.99, 0.97, "dark line: moving average of 10 log points (every 100 iters)", transform=ax.transAxes,
            ha="right", va="top", color=TEXT2, fontsize=10)
    if it.max() > unfreeze:
        unfreeze_marker(ax, unfreeze, ax.get_ylim()[1])
    fig.tight_layout(); fig.savefig(out / "1_training_loss.png", dpi=200); plt.close(fig)


def fig_lr(it, lr, out):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(it, lr, color=FT, lw=2)
    ax.set_title("Learning-rate schedule (warm-up, then cosine decay)")
    ax.set_xlabel("iteration"); ax.set_ylabel("learning rate"); kfmt(ax)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(out / "6_learning_rate.png", dpi=200); plt.close(fig)


def fig_val(hist, key, label, out, name, unfreeze):
    its = [h["iter"] for h in hist if key in h]
    ys = [h[key] for h in hist if key in h]
    pre, bic = hist[0][key], hist[0][f"{key}_bicubic"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhline(bic, color=REF, lw=2, ls="--")
    ax.axhline(pre, color=PRE, lw=2, ls="--")
    ax.plot(its, ys, color=FT, lw=2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
    x_end = max(its) if its else 1
    pad = (max(ys + [pre, bic]) - min(ys + [pre, bic])) * 0.04
    ax.text(x_end, bic - pad, f"bicubic upscaling {bic:.3f}" if key == "ssim" else f"bicubic upscaling {bic:.2f} dB",
            color=TEXT2, ha="right", va="top", fontsize=11)
    ax.text(x_end, pre - pad, f"pretrained (REDS4) {pre:.3f}" if key == "ssim" else f"pretrained (REDS4) {pre:.2f} dB",
            color=TEXT2, ha="right", va="top", fontsize=11)
    best = max(ys[1:]) if len(ys) > 1 else ys[0]
    bi = its[ys.index(best)]
    ax.annotate(f"fine-tuned {best:.3f}" if key == "ssim" else f"fine-tuned {best:.2f} dB", (bi, best),
                xytext=(0, 12), textcoords="offset points", ha="center", color=TEXT, fontweight="bold", fontsize=11)
    ax.set_title(f"Validation {label} on held-out cameras (higher is better)")
    ax.set_xlabel("iteration"); ax.set_ylabel(label); kfmt(ax)
    ax.set_ylim(min(ys + [pre, bic]) - 6 * pad, max(ys + [pre, bic]) + 6 * pad)
    if x_end > unfreeze:
        unfreeze_marker(ax, unfreeze, ax.get_ylim()[0], bottom=True)
    title = ax.get_title(loc="left"); ax.set_title("", loc="left")
    fig.suptitle(title, x=0.01, y=0.985, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, borderaxespad=0.3, handles=[plt.Line2D([], [], color=FT, lw=2, marker="o", ms=8, label="fine-tuned on MEVA"),
                       plt.Line2D([], [], color=PRE, lw=2, ls="--", label="pretrained BasicVSR++ (REDS4)"),
                       plt.Line2D([], [], color=REF, lw=2, ls="--", label="bicubic upscaling")],
)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / name, dpi=200); plt.close(fig)


def fig_bars(vals, key, label, out, name):
    names = ["Bicubic", "Pretrained\n(REDS4)", "Fine-tuned\n(MEVA)"]
    colors = [REF, PRE, FT]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor=SURF, linewidth=2)
    lo = min(vals) - (max(vals) - min(vals)) * 1.2
    ax.set_ylim(lo, max(vals) + (max(vals) - lo) * 0.12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if key == "ssim" else f"{v:.2f} dB",
                ha="center", va="bottom", color=TEXT, fontweight="bold", fontsize=12, transform=ax.transData)
    ax.set_title(f"Final {label} on held-out cameras")
    ax.set_ylabel(f"{label} (axis does not start at 0)" if lo > 0 else label); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out / name, dpi=200); plt.close(fig)


def fig_cameras(ev, out):
    cams = list(ev["finetuned"]["psnr_by_camera"])
    x = np.arange(len(cams)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.2))
    series = [("bicubic", "Bicubic", REF), ("pretrained", "Pretrained (REDS4)", PRE), ("finetuned", "Fine-tuned (MEVA)", FT)]
    allv = []
    for i, (k, lab, c) in enumerate(series):
        v = [ev[k]["psnr_by_camera"][cam] for cam in cams]; allv += v
        bars = ax.bar(x + (i - 1) * w, v, w * 0.92, color=c, label=lab, edgecolor=SURF, linewidth=2)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT2)
    lo = min(allv) - 1.5
    ax.set_ylim(lo, max(allv) + 0.8)
    ax.set_xticks(x, cams); ax.grid(axis="x", visible=False)
    ax.set_title("PSNR per held-out camera"); ax.set_ylabel("PSNR in dB (axis does not start at 0)")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout(); fig.savefig(out / "5_psnr_per_camera.png", dpi=200); plt.close(fig)


def fig_visual(a, out):
    import torch, torch.nn.functional as F
    from src.sr.basicvsrpp import build
    from training.sr.data import load_meta, _seq, _tensor
    meta = load_meta(a.data)
    val = [r for r in meta["sequences"] if r["split"] == "val"]
    picks, cams = [], set()
    for r in val:                                   # one sequence per held-out camera (that is present locally)
        if r["camera"] not in cams and Path(a.data, "val", "gt", r["id"]).is_dir():
            picks.append(r); cams.add(r["camera"])
    models = {n: build(w, "cpu") for n, w in (("pretrained", a.pretrained), ("finetuned", a.finetuned)) if Path(w).exists()}
    if len(models) < 2:
        print("visual comparison skipped: checkpoints not found"); return
    rows = []
    for r in picks[:a.visual_rows]:
        idx = range(min(r["n"], a.visual_frames))
        lq, gt = _tensor(_seq(a.data, r, "lq", idx))[None], _tensor(_seq(a.data, r, "gt", idx))[None]
        mid = lq.shape[1] // 2
        with torch.no_grad():
            outs = {n: m(lq)[0, mid].clamp(0, 1) for n, m in models.items()}
        g = gt[0, mid]
        bic = F.interpolate(lq[0, mid:mid + 1], scale_factor=4, mode="bicubic", align_corners=False)[0].clamp(0, 1)
        near = F.interpolate(lq[0, mid:mid + 1], scale_factor=4, mode="nearest")[0]
        # crop the most detailed 192x192 window of the ground truth
        e = (g.mean(0)[1:, 1:] - g.mean(0)[:-1, 1:]).abs() + (g.mean(0)[1:, 1:] - g.mean(0)[1:, :-1]).abs()
        s = 192
        dens = F.avg_pool2d(e[None, None], s, stride=32)[0, 0]
        y, x = np.unravel_index(int(dens.argmax()), dens.shape); y, x = y * 32, x * 32
        crop = lambda t: t[:, y:y + s, x:x + s].permute(1, 2, 0).numpy()
        rows.append((r["camera"], [("Low-res input", crop(near)), ("Bicubic", crop(bic)),
                                   ("Pretrained", crop(outs["pretrained"])), ("Fine-tuned", crop(outs["finetuned"])),
                                   ("Ground truth", crop(g))]))
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.5 * len(rows) + 0.6), squeeze=False)
    for i, (cam, panels) in enumerate(rows):
        for j, (t, im) in enumerate(panels):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(j == 3); sp.set_color(FT); sp.set_linewidth(3)
            if i == 0:
                ax.set_title(t, loc="center", fontsize=13, color=FT if j == 3 else TEXT)
        axes[i, 0].set_ylabel(cam, color=TEXT2, fontsize=12)
    fig.suptitle("4x super-resolution on held-out MEVA cameras (crops)", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(); fig.savefig(out / "7_visual_comparison.png", dpi=200); plt.close(fig)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(a.report))
    unfreeze = rep["args"].get("fix_flow_iters", 5000)
    hist = sorted({h["iter"]: h for h in rep["hist"]}.values(), key=lambda h: h["iter"])
    it, loss, lr = parse_log(a.log)
    if len(it):
        fig_loss(it, loss, out, unfreeze); fig_lr(it, lr, out)
    fig_val(hist, "psnr", "PSNR (dB)", out, "2_val_psnr.png", unfreeze)
    fig_val(hist, "ssim", "SSIM", out, "3_val_ssim.png", unfreeze)
    if Path(a.eval).exists():
        ev = json.load(open(a.eval))
        fig_bars([ev["bicubic"]["psnr"], ev["pretrained"]["psnr"], ev["finetuned"]["psnr"]], "psnr", "PSNR", out, "4_final_psnr.png")
        fig_bars([ev["bicubic"]["ssim"], ev["pretrained"]["ssim"], ev["finetuned"]["ssim"]], "ssim", "SSIM", out, "4b_final_ssim.png")
        fig_cameras(ev, out)
    else:
        best = max(hist[1:], key=lambda h: h["psnr"]) if len(hist) > 1 else hist[0]
        fig_bars([hist[0]["psnr_bicubic"], hist[0]["psnr"], best["psnr"]], "psnr", "PSNR", out, "4_final_psnr.png")
        fig_bars([hist[0]["ssim_bicubic"], hist[0]["ssim"], best["ssim"]], "ssim", "SSIM", out, "4b_final_ssim.png")
    if not a.no_visual and Path(a.data, "meta.json").exists():
        fig_visual(a, out)
    print("figures ->", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/logs/sr_train.log")
    ap.add_argument("--report", default="outputs/reports/sr_train.json")
    ap.add_argument("--eval", default="outputs/reports/sr_eval.json")
    ap.add_argument("--data", default="data/processed/sr_meva")
    ap.add_argument("--pretrained", default="models/pretrained/basicvsrpp_reds4.pth")
    ap.add_argument("--finetuned", default="models/finetuned/sr/basicvsrpp_meva.pth")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--visual-rows", type=int, default=2)
    ap.add_argument("--visual-frames", type=int, default=10)
    ap.add_argument("--no-visual", action="store_true")
    main(ap.parse_args())
