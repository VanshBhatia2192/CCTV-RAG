"""Datasets and metrics for BasicVSR++ fine-tuning (reads the output of training/sr/prepare_meva.py)."""
import contextlib, json, random
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from src.sr.basicvsrpp import tta_forward


def autocast(device, dtype):
    """Mixed-precision context, or a no-op when dtype is None."""
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def load_meta(root):
    return json.load(open(Path(root) / "meta.json"))


def _read(p):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _seq(root, r, sub, idx):
    d = Path(root) / r["split"] / sub / r["id"]
    return np.stack([_read(d / f"{i:05d}.png") for i in idx])


def _tensor(a):                                          # (t,h,w,3) uint8 -> (t,3,h,w) float [0,1]
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float().div_(255)


class TrainSet(torch.utils.data.Dataset):
    """Random `frames`-long windows, random lq crop of `patch` px (gt crop is 4x), flips, transpose, time reversal.

    Decoding full-size PNGs dominates CPU time, so a decoded window is reused for `reuse` random crops before a
    new one is loaded (each worker keeps its own). reuse=4 cuts decoding 4x for little loss of variety.
    """
    def __init__(s, root, frames=15, patch=64, scale=4, length=10**7, reuse=1):
        meta = load_meta(root)
        s.root, s.frames, s.patch, s.scale, s.length, s.reuse = root, frames, patch, scale, length, max(1, reuse)
        s.rows = [r for r in meta["sequences"] if r["split"] == "train" and r["n"] >= frames]
        assert s.rows, f"no train sequences with >= {frames} frames in {root}"
        s._win, s._left = None, 0

    def __len__(s):
        return s.length                              # sampling is random; length only bounds the loader

    def _window(s):
        if s._left <= 0 or s._win is None:
            r = random.choice(s.rows)
            t0 = random.randint(0, r["n"] - s.frames)
            idx = range(t0, t0 + s.frames)
            s._win, s._left = (_seq(s.root, r, "lq", idx), _seq(s.root, r, "gt", idx)), s.reuse
        s._left -= 1
        return s._win

    def __getitem__(s, _):
        lq, gt = s._window()
        h, w = lq.shape[1:3]
        y, x = random.randint(0, h - s.patch), random.randint(0, w - s.patch)
        lq = lq[:, y:y + s.patch, x:x + s.patch]
        gt = gt[:, y * s.scale:(y + s.patch) * s.scale, x * s.scale:(x + s.patch) * s.scale]
        if random.random() < 0.5:
            lq, gt = lq[:, :, ::-1], gt[:, :, ::-1]
        if random.random() < 0.5:
            lq, gt = lq[:, ::-1], gt[:, ::-1]
        if random.random() < 0.5:
            lq, gt = lq.transpose(0, 2, 1, 3), gt.transpose(0, 2, 1, 3)
        if random.random() < 0.5:
            lq, gt = lq[::-1], gt[::-1]
        return _tensor(lq), _tensor(gt)


def val_sequences(root, max_frames=30):
    """Yield (id, camera, lq, gt) for each held-out sequence, first `max_frames` frames, as (1,t,3,h,w)."""
    meta = load_meta(root)
    for r in meta["sequences"]:
        if r["split"] != "val":
            continue
        idx = range(min(r["n"], max_frames))
        yield r["id"], r["camera"], _tensor(_seq(root, r, "lq", idx))[None], _tensor(_seq(root, r, "gt", idx))[None]


def psnr(a, b, crop=0):
    """Mean per-frame RGB PSNR of (..., 3, h, w) tensors in [0,1]."""
    if crop:
        a, b = a[..., crop:-crop, crop:-crop], b[..., crop:-crop, crop:-crop]
    mse = ((a.clamp(0, 1) - b) ** 2).flatten(-3).mean(-1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()


def ssim(a, b, crop=0):
    """Mean per-frame SSIM (11x11 Gaussian, sigma 1.5) on the Y channel."""
    if crop:
        a, b = a[..., crop:-crop, crop:-crop], b[..., crop:-crop, crop:-crop]
    wts = torch.tensor([65.481, 128.553, 24.966], device=a.device).view(3, 1, 1) / 255
    ya = ((a.clamp(0, 1) * wts).sum(-3) + 16 / 255).flatten(0, -3)[:, None] * 255
    yb = ((b * wts).sum(-3) + 16 / 255).flatten(0, -3)[:, None] * 255
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None]
    f = lambda x: F.conv2d(x, k)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = f(ya), f(yb)
    va, vb, cov = f(ya * ya) - ma ** 2, f(yb * yb) - mb ** 2, f(ya * yb) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return m.mean().item()


@torch.no_grad()
def evaluate(model, root, device, max_frames=30, amp=None, tta=1):
    """PSNR/SSIM of model output and of plain bicubic upscaling, over all held-out sequences.
    tta > 1 averages that many flipped/transposed passes (self-ensemble)."""
    model.eval()
    rows = []
    for sid, cam, lq, gt in val_sequences(root, max_frames):
        lq, gt = lq.to(device), gt.to(device)
        with autocast(device, amp):
            out = tta_forward(model, lq, tta) if tta > 1 else model(lq).float()
        bic = F.interpolate(lq[0], scale_factor=4, mode="bicubic", align_corners=False)[None]
        rows.append(dict(id=sid, camera=cam, psnr=psnr(out, gt, 4), ssim=ssim(out, gt, 4),
                         psnr_bicubic=psnr(bic, gt, 4), ssim_bicubic=ssim(bic, gt, 4)))
    if not rows:
        return dict(n=0)
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 4)
    return dict(n=len(rows), psnr=mean("psnr"), ssim=mean("ssim"), psnr_bicubic=mean("psnr_bicubic"),
                ssim_bicubic=mean("ssim_bicubic"), per_seq=rows)
