"""Presentation figures for the BasicVSR++ MEVA fine-tuning run.

    python -m eval.sr.plots                      # after training + eval.sr.eval_sr
    python -m eval.sr.plots --no-visual          # charts only (no model inference)
    python -m eval.sr.plots --run v2             # v2: compares bicubic / pretrained / v1 / v2 (+ flip ensemble)

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
FT, PRE, REF, V1 = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"   # slots 1 (newest model), 2, neutral, 3
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
    if unfreeze and it.max() > unfreeze:
        unfreeze_marker(ax, unfreeze, ax.get_ylim()[1])
    fig.tight_layout(); fig.savefig(out / "1_training_loss.png", dpi=200); plt.close(fig)


def fig_lr(it, lr, out):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(it, lr, color=FT, lw=2)
    ax.set_title("Learning-rate schedule (warm-up, then cosine decay)")
    ax.set_xlabel("iteration"); ax.set_ylabel("learning rate"); kfmt(ax)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(out / "6_learning_rate.png", dpi=200); plt.close(fig)


def fig_val(hist, key, label, out, name, unfreeze, refs, model_label="fine-tuned on MEVA"):
    """refs: list of (label, value, color) horizontal reference lines."""
    its = [h["iter"] for h in hist if key in h]
    ys = [h[key] for h in hist if key in h]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    allv = ys + [v for _, v, _ in refs]
    pad = (max(allv) - min(allv)) * 0.04 or 0.01
    x_end = max(its) if its else 1
    for lab, v, c in refs:
        ax.axhline(v, color=c, lw=2, ls="--")
        ax.text(x_end, v - pad, f"{lab} {v:.3f}" if key == "ssim" else f"{lab} {v:.2f} dB", color=TEXT2, ha="right", va="top", fontsize=11)
    ax.plot(its, ys, color=FT, lw=2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
    best = max(ys[1:]) if len(ys) > 1 else ys[0]
    bi = its[ys.index(best)]
    ax.annotate(f"best {best:.3f}" if key == "ssim" else f"best {best:.2f} dB", (bi, best),
                xytext=(0, 12), textcoords="offset points", ha="center", color=TEXT, fontweight="bold", fontsize=11)
    ax.set_title(f"Validation {label} on held-out cameras (higher is better)")
    ax.set_xlabel("iteration"); ax.set_ylabel(label); kfmt(ax)
    ax.set_ylim(min(allv) - 6 * pad, max(allv) + 6 * pad)
    if unfreeze and x_end > unfreeze:
        unfreeze_marker(ax, unfreeze, ax.get_ylim()[0], bottom=True)
    title = ax.get_title(loc="left"); ax.set_title("", loc="left")
    fig.suptitle(title, x=0.01, y=0.985, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    handles = [plt.Line2D([], [], color=FT, lw=2, marker="o", ms=8, label=model_label)]
    handles += [plt.Line2D([], [], color=c, lw=2, ls="--", label=lab) for lab, _, c in refs]
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(handles), borderaxespad=0.3, handles=handles)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / name, dpi=200); plt.close(fig)


def fig_bars(items, key, label, out, name):
    """items: list of (name, value, color, hatch)."""
    names = [i[0] for i in items]; vals = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(items) + 1.5), 5.2))
    bars = ax.bar(names, vals, color=[i[2] for i in items], width=0.55, edgecolor=SURF, linewidth=2)
    for b, (_, _, c, h) in zip(bars, items):
        if h:
            b.set_hatch(h); b.set_edgecolor(SURF)
    lo = min(vals) - (max(vals) - min(vals)) * 1.2
    ax.set_ylim(lo, max(vals) + (max(vals) - lo) * 0.12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if key == "ssim" else f"{v:.2f} dB",
                ha="center", va="bottom", color=TEXT, fontweight="bold", fontsize=12)
    ax.set_title(f"Final {label} on held-out cameras")
    ax.set_ylabel(f"{label} (axis does not start at 0)" if lo > 0 else label); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(out / name, dpi=200); plt.close(fig)


def fig_cameras(ev, series, out):
    """series: list of (eval key, label, color, hatch)."""
    cams = list(ev["bicubic"]["psnr_by_camera"])
    n = len(series); x = np.arange(len(cams)); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * n + 3), 5.2))
    allv = []
    for i, (k, lab, c, h) in enumerate(series):
        v = [ev[k]["psnr_by_camera"][cam] for cam in cams]; allv += v
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=c, label=lab, edgecolor=SURF, linewidth=2, hatch=h)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10, color=TEXT2)
    ax.set_ylim(min(allv) - 1.5, max(allv) + 0.8)
    ax.set_xticks(x, cams); ax.grid(axis="x", visible=False)
    ax.set_ylabel("PSNR in dB (axis does not start at 0)")
    fig.suptitle("PSNR per held-out camera", x=0.01, y=0.985, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=n, borderaxespad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / "5_psnr_per_camera.png", dpi=200); plt.close(fig)


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
    spec = [m.split("=", 1) for m in a.visual_models] if a.visual_models else [("Pretrained", a.pretrained), ("Fine-tuned", a.finetuned)]
    models = {n: build(w, "cpu") for n, w in spec if Path(w).exists()}
    if len(models) < len(spec):
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
        rows.append((r["camera"], [("Low-res input", crop(near)), ("Bicubic", crop(bic))]
                     + [(n, crop(o)) for n, o in outs.items()] + [("Ground truth", crop(g))]))
    ncol = len(rows[0][1]); hi = ncol - 2                       # highlight the last model before the ground truth
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.2 * ncol, 3.5 * len(rows) + 0.6), squeeze=False)
    for i, (cam, panels) in enumerate(rows):
        for j, (t, im) in enumerate(panels):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(j == hi); sp.set_color(FT); sp.set_linewidth(3)
            if i == 0:
                ax.set_title(t, loc="center", fontsize=13, color=FT if j == hi else TEXT)
        axes[i, 0].set_ylabel(cam, color=TEXT2, fontsize=12)
    fig.suptitle("4x super-resolution on held-out MEVA cameras (crops)", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=TEXT)
    fig.tight_layout(); fig.savefig(out / "7_visual_comparison.png", dpi=200); plt.close(fig)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(a.report))
    unfreeze = rep["args"].get("fix_flow_iters", 5000)
    hist = sorted({h["iter"]: h for h in rep["hist"]}.values(), key=lambda h: h["iter"])
    ev = json.load(open(a.eval)) if Path(a.eval).exists() else None
    it, loss, lr = parse_log(a.log)
    if len(it):
        fig_loss(it, loss, out, unfreeze); fig_lr(it, lr, out)
    h0 = hist[0]
    if a.run == "v2":                       # v2 starts from v1: its iteration-0 score is v1 on the v2 test set
        refs = lambda k: ([("v1 (start)", h0[k], V1)] + ([("pretrained", ev["pretrained"][k], PRE)] if ev else [])
                          + [("bicubic", h0[f"{k}_bicubic"], REF)])
        mlabel = "v2 fine-tuning"
    else:
        refs = lambda k: [("pretrained (REDS4)", h0[k], PRE), ("bicubic upscaling", h0[f"{k}_bicubic"], REF)]
        mlabel = "fine-tuned on MEVA"
    fig_val(hist, "psnr", "PSNR (dB)", out, "2_val_psnr.png", unfreeze, refs("psnr"), mlabel)
    fig_val(hist, "ssim", "SSIM", out, "3_val_ssim.png", unfreeze, refs("ssim"), mlabel)
    if ev:
        style = {"bicubic": ("Bicubic", REF, None), "pretrained": ("Pretrained\n(REDS4)", PRE, None),
                 "finetuned": ("Fine-tuned\n(MEVA)", FT, None), "v1": ("v1", V1, None), "v1+tta": ("v1\n+ flip TTA", V1, "//"),
                 "v2": ("v2", FT, None), "v2+tta": ("v2\n+ flip TTA", FT, "//")}
        keys = [k for k in style if k in ev]
        for k, lab, fname in (("psnr", "PSNR", "4_final_psnr.png"), ("ssim", "SSIM", "4b_final_ssim.png")):
            fig_bars([(style[m][0], ev[m][k], style[m][1], style[m][2]) for m in keys], k, lab, out, fname)
        fig_cameras(ev, [(m, style[m][0].replace("\n", " "), style[m][1], style[m][2]) for m in keys], out)
    else:
        best = max(hist[1:], key=lambda h: h["psnr"]) if len(hist) > 1 else h0
        for k, lab, fname in (("psnr", "PSNR", "4_final_psnr.png"), ("ssim", "SSIM", "4b_final_ssim.png")):
            fig_bars([("Bicubic", h0[f"{k}_bicubic"], REF, None), ("Start", h0[k], PRE, None), ("Best", best[k], FT, None)],
                     k, lab, out, fname)
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
    ap.add_argument("--run", choices=["v1", "v2"], default="v1", help="v2: label the start as v1 and compare v1/v2")
    ap.add_argument("--visual-models", nargs="*", help="Label=path pairs for the crop comparison (default: pretrained, fine-tuned)")
    main(ap.parse_args())
