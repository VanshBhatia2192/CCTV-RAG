"""Synthetic Gait and Multi-Camera Scenario Generator for Academic Demonstration.
Generates realistic 17-keypoint 2D skeletal sequences using biomechanical walking equations.
Provides demonstration scenarios (Normal Walk, Brisk Walk, Asymmetrical Walk, Cloth-Changing Re-ID).
"""

import numpy as np
from typing import Dict, List, Tuple


def generate_walking_skeleton(
    duration_sec: float = 3.0,
    fps: float = 25.0,
    cadence_bpm: float = 110.0,
    stride_scale: float = 1.0,
    symmetry: float = 1.0,
    torso_height: float = 120.0,
    noise_std: float = 1.5,
) -> np.ndarray:
    """Generate realistic 17-keypoint COCO skeleton sequence (T, 17, 3) over time.

    Keypoints:
    0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear,
    5: l_sh, 6: r_sh, 7: l_elb, 8: r_elb, 9: l_wri, 10: r_wri,
    11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ank, 16: r_ank
    """
    T = int(duration_sec * fps)
    t = np.linspace(0, duration_sec, T)

    # Gait cycle frequency: 1 full cycle (2 steps) = cadence / 120 Hz
    freq = (cadence_bpm / 2.0) / 60.0
    phase = 2.0 * np.pi * freq * t

    kpts = np.zeros((T, 17, 3), dtype=np.float32)
    kpts[:, :, 2] = 0.95  # High confidence

    # Pelvis center
    pelvis_y = 220.0 + 4.0 * np.sin(2.0 * phase)  # Vertical bounce at 2x frequency
    pelvis_x = 160.0 + 3.0 * np.sin(phase)        # Lateral sway

    # Hips (11, 12)
    hip_width = torso_height * 0.28
    kpts[:, 11, 0] = pelvis_x - hip_width / 2.0
    kpts[:, 11, 1] = pelvis_y
    kpts[:, 12, 0] = pelvis_x + hip_width / 2.0
    kpts[:, 12, 1] = pelvis_y

    # Shoulders (5, 6)
    sh_y = pelvis_y - torso_height
    sh_width = torso_height * 0.42
    sh_sway = -3.0 * np.sin(phase)  # Counter-phase sway
    kpts[:, 5, 0] = pelvis_x - sh_width / 2.0 + sh_sway
    kpts[:, 5, 1] = sh_y
    kpts[:, 6, 0] = pelvis_x + sh_width / 2.0 + sh_sway
    kpts[:, 6, 1] = sh_y

    # Head (0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear)
    head_y = sh_y - torso_height * 0.28
    kpts[:, 0, 0] = pelvis_x + sh_sway * 0.7
    kpts[:, 0, 1] = head_y
    kpts[:, 1, 0] = pelvis_x - 10.0 + sh_sway * 0.7
    kpts[:, 1, 1] = head_y - 6.0
    kpts[:, 2, 0] = pelvis_x + 10.0 + sh_sway * 0.7
    kpts[:, 2, 1] = head_y - 6.0
    kpts[:, 3, 0] = pelvis_x - 20.0 + sh_sway * 0.7
    kpts[:, 3, 1] = head_y - 4.0
    kpts[:, 4, 0] = pelvis_x + 20.0 + sh_sway * 0.7
    kpts[:, 4, 1] = head_y - 4.0

    # Leg kinematics
    thigh_len = torso_height * 0.75
    shin_len = torso_height * 0.75
    leg_span = torso_height * 0.45 * stride_scale

    # Left leg phase
    l_swing = np.sin(phase)
    l_lift = np.maximum(0.0, -np.sin(phase)) ** 1.5 * 25.0

    # Right leg phase (phase shifted by pi, scaled by symmetry)
    r_phase = phase + np.pi
    r_swing = np.sin(r_phase) * symmetry
    r_lift = np.maximum(0.0, -np.sin(r_phase)) ** 1.5 * 25.0 * symmetry

    # Left knee & ankle (13, 15)
    kpts[:, 15, 0] = kpts[:, 11, 0] + leg_span * l_swing
    kpts[:, 15, 1] = pelvis_y + thigh_len + shin_len - l_lift
    kpts[:, 13, 0] = (kpts[:, 11, 0] + kpts[:, 15, 0]) / 2.0 + 6.0 * np.maximum(0.0, -l_swing)
    kpts[:, 13, 1] = pelvis_y + thigh_len - l_lift * 0.6

    # Right knee & ankle (14, 16)
    kpts[:, 16, 0] = kpts[:, 12, 0] + leg_span * r_swing
    kpts[:, 16, 1] = pelvis_y + thigh_len + shin_len - r_lift
    kpts[:, 14, 0] = (kpts[:, 12, 0] + kpts[:, 16, 0]) / 2.0 + 6.0 * np.maximum(0.0, -r_swing)
    kpts[:, 14, 1] = pelvis_y + thigh_len - r_lift * 0.6

    # Arm kinematics (counter-phase to legs)
    arm_span = torso_height * 0.3 * stride_scale
    arm_len = torso_height * 0.45

    # Left arm (7: elbow, 9: wrist) - in phase with right leg
    kpts[:, 9, 0] = kpts[:, 5, 0] - arm_span * r_swing
    kpts[:, 9, 1] = sh_y + arm_len * 1.8
    kpts[:, 7, 0] = (kpts[:, 5, 0] + kpts[:, 9, 0]) / 2.0
    kpts[:, 7, 1] = sh_y + arm_len

    # Right arm (8: elbow, 10: wrist) - in phase with left leg
    kpts[:, 10, 0] = kpts[:, 6, 0] - arm_span * l_swing
    kpts[:, 10, 1] = sh_y + arm_len * 1.8
    kpts[:, 8, 0] = (kpts[:, 6, 0] + kpts[:, 10, 0]) / 2.0
    kpts[:, 8, 1] = sh_y + arm_len

    # Add realistic sensor jitter
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, kpts[:, :, :2].shape).astype(np.float32)
        kpts[:, :, :2] += noise

    return kpts


DEMO_SCENARIOS = {
    "casual_walk": {
        "title": "Scenario A: Regular Casual Walk",
        "description": "Standard adult walking gait at normal pace (~105 steps/min) with symmetric stride.",
        "params": dict(duration_sec=3.0, cadence_bpm=106.0, stride_scale=1.0, symmetry=1.0),
        "app_color": "#2563eb",  # Blue
        "app_label": "Navy Blue Jacket & Jeans",
        "cam": "Cam 01 (Entrance)",
    },
    "brisk_walk": {
        "title": "Scenario B: Fast Paced / Hurried Walk",
        "description": "Brisk walking pace (~135 steps/min) with wider leg stride span and higher knee flexion.",
        "params": dict(duration_sec=3.0, cadence_bpm=136.0, stride_scale=1.35, symmetry=0.98),
        "app_color": "#059669",  # Green
        "app_label": "Green Parka & Cargo Pants",
        "cam": "Cam 02 (Hallway)",
    },
    "limp_asymmetric": {
        "title": "Scenario C: Pathological / Injured Gait",
        "description": "Asymmetrical limp (0.64 symmetry) with reduced right-leg stride length and pronounced body sway.",
        "params": dict(duration_sec=3.0, cadence_bpm=92.0, stride_scale=0.85, symmetry=0.64),
        "app_color": "#d97706",  # Amber
        "app_label": "Brown Coat & Gray Slacks",
        "cam": "Cam 03 (Terrace)",
    },
}
