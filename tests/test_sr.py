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
