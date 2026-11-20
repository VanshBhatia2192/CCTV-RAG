"""Unit tests for 2D Skeleton-based Pose-Guided Gait Biometrics and ST-GCN network."""

import numpy as np
import pytest
import torch

from src.gait import pose_gait as pg


def test_adjacency_matrix():
    """Verify 17-keypoint anatomical skeleton graph structure."""
    A = pg.build_adjacency_matrix(17, pg.COCO_PAIRS)
    assert A.shape == (17, 17)
    # Self-loops should exist
    assert torch.all(torch.diag(A) > 0)
    # Adjacency should be symmetric
    assert torch.allclose(A, A.T, atol=1e-5)


def test_skeleton_normalization():
    """Verify coordinate translation (centering on pelvis) and torso scaling."""
    T = 20
    # Synthetic keypoints: (T, 17, 3)
    kpts = np.zeros((T, 17, 3), dtype=np.float32)
    # Set pelvis (left hip: 11, right hip: 12) around (100, 200)
    kpts[:, 11, 0] = 95.0
    kpts[:, 11, 1] = 200.0
    kpts[:, 11, 2] = 0.9
    kpts[:, 12, 0] = 105.0
    kpts[:, 12, 1] = 200.0
    kpts[:, 12, 2] = 0.9

    # Set shoulders (5, 6) around (100, 100) -> torso length = 100
    kpts[:, 5, 0] = 90.0
    kpts[:, 5, 1] = 100.0
    kpts[:, 5, 2] = 0.9
    kpts[:, 6, 0] = 110.0
    kpts[:, 6, 1] = 100.0
    kpts[:, 6, 2] = 0.9

    norm = pg.normalize_skeleton_sequence(kpts)
    assert norm.shape == (3, T, 17)
    # Pelvis midpoint should be centered at (0, 0)
    pelvis_norm = (norm[:2, :, 11] + norm[:2, :, 12]) / 2.0
    assert np.allclose(pelvis_norm, 0.0, atol=1e-4)


def test_kinematic_gait_metrics():
    """Verify gait cadence, step length, knee flexion, and symmetry computation."""
    T = 50
    kpts = np.zeros((T, 17, 3), dtype=np.float32)
    # Generate oscillating ankle stride pattern at 2 Hz
    t = np.linspace(0, 2.0, T)
    la_x = 0.3 * np.sin(2 * np.pi * 2.0 * t)
    ra_x = -0.3 * np.sin(2 * np.pi * 2.0 * t)

    kpts[:, 15, 0] = la_x  # left ankle
    kpts[:, 15, 1] = 1.0
    kpts[:, 16, 0] = ra_x  # right ankle
    kpts[:, 16, 1] = 1.0

    # Hips & knees
    kpts[:, 11, 1] = 0.5  # l_hip
    kpts[:, 12, 1] = 0.5  # r_hip
    kpts[:, 13, 1] = 0.75  # l_knee
    kpts[:, 14, 1] = 0.75  # r_knee

    metrics = pg.compute_kinematic_gait_metrics(kpts, fps=25.0)
    assert "cadence_bpm" in metrics
    assert "step_length_norm" in metrics
    assert "knee_flexion_range_deg" in metrics
    assert "gait_symmetry" in metrics
    assert "body_sway" in metrics
    assert metrics["step_length_norm"] > 0


def test_stgcn_forward_pass():
    """Verify ST-GCN network forward pass and output embedding normalization."""
    model = pg.STGCNPoseGait(in_channels=3, out_dim=256)
    model.eval()

    B, C, T, V = 2, 3, 32, 17
    x = torch.randn(B, C, T, V)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, 256)
    # Check L2 unit normalization
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)
