"""Tests for the BasicVSR++ super-resolution module (src/sr) and its training data (training/sr)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sr import basicvsrpp as B


def test_flow_warp_zero_flow_is_identity():
    x = torch.rand(2, 3, 16, 20)
    assert torch.allclose(B.flow_warp(x, torch.zeros(2, 16, 20, 2)), x, atol=1e-5)


def test_forward_shape():
    m = B.BasicVSRPlusPlus().eval()
    with torch.no_grad():
        y = m(torch.rand(1, 3, 3, 64, 72))
    assert y.shape == (1, 3, 3, 256, 288)


def test_load_mmedit_style_checkpoint(tmp_path):
    m = B.BasicVSRPlusPlus()
    ck = tmp_path / "mmedit.pth"
    torch.save({"state_dict": {f"generator.{k}": v for k, v in m.state_dict().items()}, "meta": {}}, ck)
    m2 = B.load_weights(B.BasicVSRPlusPlus(), ck)
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k


def test_load_training_checkpoint(tmp_path):
    m = B.BasicVSRPlusPlus()
    ck = tmp_path / "ft.pth"
    torch.save({"model": m.state_dict(), "iter": 5}, ck)
    B.load_weights(B.BasicVSRPlusPlus(), ck)


@pytest.mark.parametrize("t", [1, 2, 9, 10, 23])
def test_upscale_windows_keep_every_frame_in_order(monkeypatch, t):
    from src.sr import enhance as E

    class Fake(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.p = torch.nn.Parameter(torch.zeros(1))

        def forward(s, x):
            return torch.nn.functional.interpolate(x[0], scale_factor=4)[None]

    monkeypatch.setattr(E, "_MODEL", Fake())
    frames = (np.arange(t)[:, None, None, None] * np.ones((1, 64, 64, 3))).astype(np.uint8)
    out = E.upscale(frames, window=10, overlap=4)
    assert out.shape == (t, 256, 256, 3)
    assert list(out[:, 0, 0, 0]) == list(range(t))


def test_metrics():
    from training.sr.data import psnr, ssim
    a = torch.rand(1, 2, 3, 32, 32)
    assert psnr(a, a) > 90
    assert abs(ssim(a, a) - 1) < 1e-4
    assert ssim(a, torch.rand_like(a)) < 0.5


def test_tta_undoes_its_flips():
    class Up(torch.nn.Module):          # position-dependent but flip-equivariant: plain 4x nearest upsample
        def forward(s, x):
            return torch.nn.functional.interpolate(x.flatten(0, 1), scale_factor=4).view(*x.shape[:3], x.shape[3] * 4, x.shape[4] * 4)
    x = torch.rand(1, 3, 3, 64, 72)
    expect = Up()(x)
    for n in (2, 4, 8):
        assert torch.allclose(B.tta_forward(Up(), x, n), expect, atol=1e-6), n


def test_ema_update_moves_towards_model():
    from training.sr.train import ema_update
    m, e = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    with torch.no_grad():
        for p in e.parameters():
            p.zero_()
        for p in m.parameters():
            p.fill_(1.0)
    ema_update(e, m, decay=0.9, step=10**6)
    assert all(torch.allclose(p, torch.full_like(p, 0.1)) for p in e.parameters())


def test_trainset_reuse_decodes_less(tmp_path, monkeypatch):
    import json
    from training.sr import data as D
    json.dump({"sequences": [{"id": "a", "split": "train", "camera": "c", "n": 6}]}, open(tmp_path / "meta.json", "w"))
    calls = []
    monkeypatch.setattr(D, "_seq", lambda root, r, sub, idx: calls.append(sub) or np.zeros((len(idx), 80 if sub == "lq" else 320, 80 if sub == "lq" else 320, 3), np.uint8))
    ds = D.TrainSet(tmp_path, frames=4, patch=64, reuse=4)
    shapes = [ds[0][0].shape for _ in range(8)]
    assert len(calls) == 4 and all(sh == (4, 3, 64, 64) for sh in shapes)   # 8 samples, 2 decodes of (lq, gt)
