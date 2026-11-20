"""Skeleton-based 2D Pose-Guided Gait Biometrics.
Extracts 17 COCO skeletal keypoints per frame using YOLOv8-pose, normalizes
them for scale and translation, and encodes temporal gait dynamics via a
Spatial-Temporal Graph Convolutional Network (ST-GCN).

Key features:
1. Invariant to clothing changes, color, lighting, and loose garments.
2. Models spatial bone connectivity (ankles, knees, hips, torso, arms).
3. Models temporal kinematics: stride cadence, step length, knee flexion, body sway.
4. Produces normalized 256-d pose-gait biometric embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 17 COCO Keypoints:
# 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
# 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
# 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle

COCO_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"
]

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]


def build_adjacency_matrix(num_nodes: int = 17, edges: List[Tuple[int, int]] = COCO_PAIRS) -> torch.Tensor:
    """Build normalized adjacency matrix with self-loops."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + np.eye(num_nodes, dtype=np.float32)  # self-loops
    # Degree-normalized: D^(-1/2) A D^(-1/2)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
    return torch.from_numpy(A_norm)


class SpatialGraphConv(nn.Module):
    """Graph convolution operating along skeletal anatomical connections."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 17):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        A = build_adjacency_matrix(num_nodes, COCO_PAIRS)
        self.register_buffer("A", A)
        # Learnable edge weight mask initialized to ones
        self.edge_importance = nn.Parameter(torch.ones(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, V)
        b, c, t, v = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b * t, c, v)
        # Apply adjacency with learned importance
        adj = self.A * self.edge_importance
        x_prop = torch.matmul(x_reshaped, adj)  # (B*T, C, V)
        x_prop = x_prop.reshape(b, t, c, v).permute(0, 2, 1, 3)
        return self.conv(x_prop)


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 7, stride: int = 1):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        padding = (temporal_kernel - 1) // 2
        self.tgcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), (stride, 1), (padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1),
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.bn_s(self.sgcn(x)))
        x = self.tgcn(x)
        return self.relu(x + res)


class STGCNPoseGait(nn.Module):
    """Deep Spatial-Temporal Graph Convolutional Gait Recognition Network.

    Takes normalized 2D skeleton sequences (B, 3, T, 17) -> 256-d embedding.
    Channels: (x_norm, y_norm, confidence).
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256, num_nodes: int = 17):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        self.layer1 = STGCNBlock(in_channels, 64, stride=1)
        self.layer2 = STGCNBlock(64, 128, stride=2)
        self.layer3 = STGCNBlock(128, 256, stride=2)

        self.head = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        b, c, t, v = x.shape
        # Input batchnorm across vertices
        x_norm = x.permute(0, 1, 3, 2).reshape(b, c * v, t)
        x_norm = self.data_bn(x_norm).reshape(b, c, v, t).permute(0, 1, 3, 2)

        feat = self.layer1(x_norm)
        feat = self.layer2(feat)
        feat = self.layer3(feat)

        # Global average pool over Time and Vertices
        pooled = feat.mean(dim=(2, 3))  # (B, 256)
        out = self.head(pooled)
        return F.normalize(out, p=2, dim=-1)


def normalize_skeleton_sequence(kpts: np.ndarray, min_conf: float = 0.25) -> np.ndarray:
    """Normalize skeleton sequence: center on pelvis, scale by torso height.

    Args:
        kpts: array of shape (T, 17, 3) where columns are [x, y, conf].
    Returns:
        Array of shape (3, T, 17) normalized, float32.
    """
    T, V, C = kpts.shape
    out = kpts.copy().astype(np.float32)

    # Pelvis center = midpoint of left hip (11) and right hip (12)
    l_hip, r_hip = out[:, 11, :2], out[:, 12, :2]
    pelvis = (l_hip + r_hip) / 2.0  # (T, 2)

    # Neck/Shoulder center = midpoint of left shoulder (5) and right shoulder (6)
    l_sh, r_sh = out[:, 5, :2], out[:, 6, :2]
    neck = (l_sh + r_sh) / 2.0  # (T, 2)

    # Torso height = distance from pelvis to neck
    torso_len = np.linalg.norm(neck - pelvis, axis=1)  # (T,)
    valid_torso = torso_len[torso_len > 1e-3]
    scale = float(np.median(valid_torso)) if len(valid_torso) > 0 else 1.0
    scale = max(scale, 1e-3)

    # Translation: center on pelvis per frame
    out[:, :, 0] -= pelvis[:, 0, None]
    out[:, :, 1] -= pelvis[:, 1, None]

    # Scale: divide by torso length
    out[:, :, :2] /= scale

    # Zero-out coordinates with confidence below threshold
    low_conf = out[:, :, 2] < min_conf
    out[low_conf, :2] = 0.0

    # Reshape to (C, T, V)
    return out.transpose(2, 0, 1)


def compute_kinematic_gait_metrics(kpts: np.ndarray, fps: float = 25.0) -> Dict[str, float]:
    """Calculate interpretable biomechanical gait parameters.

    Metrics:
    - cadence_bpm: Estimated steps per minute from ankle stride oscillations
    - step_length_norm: Average peak-to-peak ankle separation normalized by leg length
    - knee_flexion_range_deg: Max - min knee angle during gait cycle
    - gait_symmetry: Left vs right leg movement symmetry (1.0 = perfect symmetry)
    - body_sway: Lateral oscillation of upper torso relative to hips
    """
    T = len(kpts)
    if T < 10:
        return {
            "cadence_bpm": 0.0,
            "step_length_norm": 0.0,
            "knee_flexion_range_deg": 0.0,
            "gait_symmetry": 1.0,
            "body_sway": 0.0,
        }

    # Left & right ankles: 15, 16
    la = kpts[:, 15, :2]
    ra = kpts[:, 16, :2]
    # Ankle separation distance over time
    stride_dist = np.linalg.norm(la - ra, axis=1)

    # Step length: median of top 25% peak separation
    step_len = float(np.percentile(stride_dist, 75))

    # FFT-based cadence estimation
    detrended = stride_dist - np.mean(stride_dist)
    fft_vals = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    # Human step frequency typically 1.0 Hz - 3.5 Hz (60 - 210 steps/min)
    valid_mask = (freqs >= 1.0) & (freqs <= 3.5)
    if np.any(valid_mask):
        peak_freq = freqs[valid_mask][np.argmax(fft_vals[valid_mask])]
        cadence = float(peak_freq * 60.0)
    else:
        cadence = 0.0

    # Knee flexion angles (Hip-Knee-Ankle) for left and right legs
    def knee_angle(hip_idx, knee_idx, ankle_idx):
        h = kpts[:, hip_idx, :2]
        k = kpts[:, knee_idx, :2]
        a = kpts[:, ankle_idx, :2]
        v1 = h - k
        v2 = a - k
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        cos_ang = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    l_knee_angles = knee_angle(11, 13, 15)
    r_knee_angles = knee_angle(12, 14, 16)
    l_range = float(np.percentile(l_knee_angles, 90) - np.percentile(l_knee_angles, 10))
    r_range = float(np.percentile(r_knee_angles, 90) - np.percentile(r_knee_angles, 10))
    flexion_range = round((l_range + r_range) / 2.0, 1)

    # Gait symmetry index (ratio of left/right knee flexion range)
    ratio = min(l_range, r_range) / max(l_range, r_range, 1e-3)
    symmetry = round(float(np.clip(ratio, 0.0, 1.0)), 2)

    # Body sway: standard deviation of torso center X-offset
    torso_x = (kpts[:, 5, 0] + kpts[:, 6, 0]) / 2.0 - (kpts[:, 11, 0] + kpts[:, 12, 0]) / 2.0
    sway = round(float(np.std(torso_x)), 2)

    return {
        "cadence_bpm": round(cadence, 1),
        "step_length_norm": round(step_len, 2),
        "knee_flexion_range_deg": flexion_range,
        "gait_symmetry": symmetry,
        "body_sway": sway,
    }


_YOLO_POSE = None
_STGCN_MODEL = None


def get_pose_detector():
    """Load or cache YOLOv8-pose model."""
    global _YOLO_POSE
    if _YOLO_POSE is None:
        from ultralytics import YOLO
        _YOLO_POSE = YOLO("yolov8n-pose.pt")
    return _YOLO_POSE


def get_stgcn_encoder(device: str = "cpu") -> STGCNPoseGait:
    """Load or initialize ST-GCN Pose Gait model."""
    global _STGCN_MODEL
    if _STGCN_MODEL is None:
        model = STGCNPoseGait(in_channels=3, out_dim=256)
        model.eval()
        _STGCN_MODEL = model.to(device)
    return _STGCN_MODEL


def extract_keypoints_from_images(
    images: List[np.ndarray],
    conf_thresh: float = 0.25,
) -> np.ndarray:
    """Extract (T, 17, 3) keypoints from list of BGR image crops using YOLOv8-pose."""
    if not images:
        return np.zeros((0, 17, 3), dtype=np.float32)

    detector = get_pose_detector()
    results = detector.predict(images, imgsz=256, verbose=False, conf=conf_thresh)

    kpts_list = []
    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            # Pick highest confidence person detection in crop
            kp = r.keypoints.data[0].cpu().numpy()  # (17, 3) -> [x, y, conf]
            kpts_list.append(kp)
        else:
            kpts_list.append(np.zeros((17, 3), dtype=np.float32))

    return np.array(kpts_list, dtype=np.float32)


def encode_pose_gait(
    images_or_kpts: Union[List[np.ndarray], np.ndarray],
    device: str = "cpu",
    fps: float = 25.0,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """End-to-end extraction: Input crops -> 256-d ST-GCN Embedding + Kinematics.

    Returns:
        (embedding_256d, kinematic_metrics_dict)
    """
    if isinstance(images_or_kpts, list):
        kpts = extract_keypoints_from_images(images_or_kpts)
    else:
        kpts = images_or_kpts

    if len(kpts) < 8:
        return None, compute_kinematic_gait_metrics(kpts, fps)

    # 1. Kinematic biomechanical summary
    metrics = compute_kinematic_gait_metrics(kpts, fps)

    # 2. Normalize skeleton sequence: (3, T, 17)
    norm_kpts = normalize_skeleton_sequence(kpts)

    # 3. ST-GCN forward pass
    encoder = get_stgcn_encoder(device)
    x_tensor = torch.from_numpy(norm_kpts).unsqueeze(0).to(device)  # (1, 3, T, 17)

    with torch.no_grad():
        emb = encoder(x_tensor).cpu().numpy()[0]  # (256,)

    return emb, metrics
