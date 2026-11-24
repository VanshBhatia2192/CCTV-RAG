"""Skeleton (2D pose) gait biometrics.

Pipeline: person crops -> 17 COCO keypoints per frame (ViTPose, top-down on the crop) ->
normalised skeleton sequence (pelvis-centred, torso-scaled, plus per-frame velocity) ->
ST-GCN encoder trained on MEVID (training/gait/train_pose.py) -> L2-normalised 256-d embedding
(compare with cosine similarity).

The encoder only produces embeddings when trained weights exist (models/finetuned/gait/stgcn_pose_mevid.pt,
or MODELS["gait_pose"]["weights"]); without them `encode_pose_gait` returns None instead of a meaningless
random-network vector.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 17 COCO keypoints:
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
# 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
COCO_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left <-> right
IN_CHANNELS = 5          # x, y, dx, dy, confidence
MIN_FRAMES = 16          # shortest sequence that gets an embedding
DEFAULT_WEIGHTS = "models/finetuned/gait/stgcn_pose_mevid.pt"
DEFAULT_POSE_MODEL = "usyd-community/vitpose-base-simple"


# ----------------------------------------------------------------------------- model

def build_adjacency_matrix(num_nodes: int = 17, edges: List[Tuple[int, int]] = COCO_PAIRS) -> torch.Tensor:
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    A = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    d = 1.0 / np.sqrt(A.sum(1))
    return torch.from_numpy(d[:, None] * A * d[None, :])


class SpatialGraphConv(nn.Module):
    """Graph convolution along the skeleton, with a learnable edge-importance mask."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 17):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.register_buffer("A", build_adjacency_matrix(num_nodes))
        self.edge_importance = nn.Parameter(torch.ones(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x: (B, C, T, V)
        return self.conv(torch.einsum("bctv,vw->bctw", x, self.A * self.edge_importance))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels)
        self.bn_s = nn.BatchNorm2d(out_channels)
        pad = (temporal_kernel - 1) // 2
        self.tgcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, (stride, 1)), nn.BatchNorm2d(out_channels))
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = F.relu(self.bn_s(self.sgcn(x)))
        return F.relu(self.tgcn(x) + res)


class STGCNPoseGait(nn.Module):
    """(B, 5, T, 17) skeleton sequence -> L2-normalised embedding (B, out_dim).

    Trained with a BNNeck: `forward(x, return_logits=True)` also returns identity logits
    (only when built with num_classes > 0)."""

    def __init__(self, in_channels: int = IN_CHANNELS, out_dim: int = 256, num_nodes: int = 17,
                 widths: Tuple[int, ...] = (64, 64, 128, 128, 256, 256), strides: Tuple[int, ...] = (1, 1, 2, 1, 2, 1),
                 num_classes: int = 0, dropout: float = 0.1):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        blocks, c = [], in_channels
        for w, s in zip(widths, strides):
            blocks.append(STGCNBlock(c, w, stride=s, dropout=dropout)); c = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(c, out_dim)
        self.bnneck = nn.BatchNorm1d(out_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(out_dim, num_classes, bias=False) if num_classes > 0 else None

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        b, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * v, t)).reshape(b, c, v, t).permute(0, 1, 3, 2)
        feat = self.blocks(x).mean(dim=(2, 3))
        raw = self.fc(feat)                     # metric-loss feature
        bn = self.bnneck(raw)
        emb = F.normalize(bn, dim=-1)
        if return_logits:
            return emb, raw, (self.classifier(bn) if self.classifier is not None else None)
        return emb


# ----------------------------------------------------------------------------- skeleton features

def normalize_skeleton_sequence(kpts: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """(T, 17, 3) [x, y, conf] in pixels -> (5, T, 17) float32: x, y (pelvis-centred, torso-scaled), dx, dy, conf.

    Low-confidence joints get zero coordinates/velocity and keep their (low) confidence, so the network
    can learn to ignore them."""
    k = np.asarray(kpts, dtype=np.float32)
    T = len(k)
    xy, conf = k[:, :, :2].copy(), k[:, :, 2].copy()
    ok = conf >= min_conf
    hips_ok = ok[:, 11] & ok[:, 12]
    pelvis = (xy[:, 11] + xy[:, 12]) / 2.0
    if hips_ok.any():   # frames with unreliable hips borrow the nearest reliable pelvis
        idx = np.where(hips_ok)[0]
        near = idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(1)]
        pelvis = pelvis[near]
    else:
        pelvis = np.repeat(np.nanmedian(np.where(ok[..., None], xy, np.nan), axis=(0, 1))[None], T, 0)
        pelvis = np.nan_to_num(pelvis)
    neck = (xy[:, 5] + xy[:, 6]) / 2.0
    torso = np.linalg.norm(neck - pelvis, axis=1)
    good = torso[hips_ok & ok[:, 5] & ok[:, 6]]
    scale = float(np.median(good)) if len(good) else float(np.median(torso[torso > 1e-3])) if (torso > 1e-3).any() else 1.0
    xy = (xy - pelvis[:, None]) / max(scale, 1e-3)
    xy[~ok] = 0.0
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    vel[1:][~(ok[1:] & ok[:-1])] = 0.0
    out = np.concatenate([xy, vel, conf[..., None]], axis=2)       # (T, 17, 5)
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def flip_keypoints(kpts: np.ndarray, width: float) -> np.ndarray:
    """Mirror a (T, 17, 3) pixel-space sequence horizontally (swaps left/right joints)."""
    out = kpts[:, FLIP_PERM].copy()
    out[:, :, 0] = width - out[:, :, 0]
    return out


def compute_kinematic_gait_metrics(kpts: np.ndarray, fps: float = 25.0) -> Dict[str, float]:
    """Interpretable gait descriptors (for display only; retrieval uses the embedding):
    cadence_bpm, step_length_norm (ankle separation / torso), knee_flexion_range_deg, gait_symmetry, body_sway."""
    kpts = np.asarray(kpts, dtype=np.float32)
    T = len(kpts)
    if T < 10:
        return {"cadence_bpm": 0.0, "step_length_norm": 0.0, "knee_flexion_range_deg": 0.0, "gait_symmetry": 1.0, "body_sway": 0.0}
    pelvis = (kpts[:, 11, :2] + kpts[:, 12, :2]) / 2.0
    neck = (kpts[:, 5, :2] + kpts[:, 6, :2]) / 2.0
    torso = float(np.median(np.linalg.norm(neck - pelvis, axis=1))) or 1.0
    stride = np.linalg.norm(kpts[:, 15, :2] - kpts[:, 16, :2], axis=1)
    step_len = float(np.percentile(stride, 75)) / torso
    det = stride - stride.mean()
    fft = np.abs(np.fft.rfft(det)); freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    band = (freqs >= 1.0) & (freqs <= 3.5)      # ankle separation peaks twice per stride: 60-210 steps/min
    cadence = float(freqs[band][np.argmax(fft[band])] * 60.0) if band.any() else 0.0

    def knee_angle(h, k, a):
        v1, v2 = kpts[:, h, :2] - kpts[:, k, :2], kpts[:, a, :2] - kpts[:, k, :2]
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    lr = [float(np.percentile(x, 90) - np.percentile(x, 10)) for x in (knee_angle(11, 13, 15), knee_angle(12, 14, 16))]
    sway = (kpts[:, 5, 0] + kpts[:, 6, 0]) / 2.0 - pelvis[:, 0]
    return {
        "cadence_bpm": round(cadence, 1),
        "step_length_norm": round(step_len, 2),
        "knee_flexion_range_deg": round(sum(lr) / 2.0, 1),
        "gait_symmetry": round(float(np.clip(min(lr) / max(max(lr), 1e-3), 0, 1)), 2),
        "body_sway": round(float(np.std(sway)) / torso, 3),
    }


# ----------------------------------------------------------------------------- keypoint extraction

_CROP_POSE = None      # (processor, model, device) for ViTPose
_YOLO_POSE = None


def _pose_model_name() -> str:
    try:
        from src.common.config import MODELS
        return MODELS.get("gait_pose", {}).get("keypoints", DEFAULT_POSE_MODEL)
    except Exception:
        return DEFAULT_POSE_MODEL


def get_crop_pose_model(device: Optional[str] = None):
    """ViTPose (top-down): each image is one person crop and the whole crop is the person box."""
    global _CROP_POSE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _CROP_POSE is None or _CROP_POSE[2] != device:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        name = _pose_model_name()
        proc = AutoProcessor.from_pretrained(name)
        model = VitPoseForPoseEstimation.from_pretrained(name).to(device).eval()
        if device.startswith("cuda"):
            model = model.half()
        _CROP_POSE = (proc, model, device)
    return _CROP_POSE


def get_pose_detector():
    """YOLOv8-pose for full frames (detects and poses several people at once; used by demo_showcase)."""
    global _YOLO_POSE
    if _YOLO_POSE is None:
        from ultralytics import YOLO
        p = Path("models/pretrained/yolov8m-pose.pt")
        _YOLO_POSE = YOLO(str(p) if p.exists() else "yolov8m-pose.pt")
    return _YOLO_POSE


@torch.no_grad()
def extract_keypoints_from_images(images: List[np.ndarray], device: Optional[str] = None, batch: int = 64,
                                  bgr: bool = True) -> np.ndarray:
    """Person crops (H, W, 3 uint8; BGR by default) -> (T, 17, 3) keypoints [x, y, score] in crop pixels."""
    if not len(images):
        return np.zeros((0, 17, 3), dtype=np.float32)
    proc, model, dev = get_crop_pose_model(device)
    out = []
    for i in range(0, len(images), batch):
        ims = [im[:, :, ::-1] if bgr else im for im in images[i:i + batch]]
        boxes = [[[0.0, 0.0, float(im.shape[1]), float(im.shape[0])]] for im in ims]
        inp = proc(ims, boxes=boxes, return_tensors="pt").to(dev)
        if dev.startswith("cuda"):
            inp["pixel_values"] = inp["pixel_values"].half()
        o = model(**inp); o.heatmaps = o.heatmaps.float()   # post-processing (scipy) needs float32
        res = proc.post_process_pose_estimation(o, boxes=boxes)
        for r in res:
            out.append(np.concatenate([r[0]["keypoints"].float().cpu().numpy(), r[0]["scores"].float().cpu().numpy()[:, None]], 1))
    return np.asarray(out, dtype=np.float32)


# ----------------------------------------------------------------------------- encoder

_ENCODER = None    # (model, device, meta)


def _weights_path() -> Path:
    try:
        from src.common.config import MODELS
        return Path(MODELS.get("gait_pose", {}).get("weights", DEFAULT_WEIGHTS))
    except Exception:
        return Path(DEFAULT_WEIGHTS)


def load_encoder(device: str = "cpu", ckpt: Optional[str] = None):
    """Trained ST-GCN encoder, or None (with a warning) if no checkpoint is found."""
    global _ENCODER
    path = Path(ckpt) if ckpt else _weights_path()
    if _ENCODER is not None and _ENCODER[1] == device and _ENCODER[2].get("path") == str(path):
        return _ENCODER[0]
    if not path.exists():
        print(f"[pose_gait] no trained ST-GCN weights at {path}: skeleton gait embeddings are disabled "
              "(train with training/gait/train_pose.py)", flush=True)
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("model_cfg", {})
    model = STGCNPoseGait(**cfg)
    model.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("classifier.")}, strict=False)
    model.eval().to(device)
    _ENCODER = (model, device, dict(path=str(path), seq_len=int(ck.get("seq_len", 64)), min_conf=float(ck.get("min_conf", 0.3))))
    return model


def get_stgcn_encoder(device: str = "cpu"):
    return load_encoder(device)


@torch.no_grad()
def embed_keypoints(kpts: np.ndarray, device: str = "cpu", model=None, seq_len: Optional[int] = None,
                    min_conf: Optional[float] = None) -> Optional[np.ndarray]:
    """(T, 17, 3) keypoints of one tracklet -> 256-d embedding (mean over sliding windows), or None."""
    if model is None:
        model = load_encoder(device)
        if model is None:
            return None
    meta = _ENCODER[2] if _ENCODER is not None and _ENCODER[0] is model else {}
    L = seq_len or meta.get("seq_len", 64)
    mc = min_conf if min_conf is not None else meta.get("min_conf", 0.3)
    kpts = np.asarray(kpts, dtype=np.float32)
    if len(kpts) < MIN_FRAMES:
        return None
    x = normalize_skeleton_sequence(kpts, mc)                 # (5, T, 17)
    T = x.shape[1]
    starts = [0] if T <= L else list(range(0, T - L + 1, max(1, L // 2)))
    if T > L and starts[-1] != T - L:
        starts.append(T - L)
    wins = np.stack([x[:, s:s + L] for s in starts])
    dev = next(model.parameters()).device
    e = model(torch.from_numpy(wins).to(dev)).float().mean(0)
    return F.normalize(e, dim=0).cpu().numpy()


def encode_pose_gait(images_or_kpts: Union[List[np.ndarray], np.ndarray], device: str = "cpu",
                     fps: float = 25.0) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Consecutive person crops (list of BGR arrays) or (T, 17, 3) keypoints -> (embedding or None, kinematics)."""
    kpts = extract_keypoints_from_images(images_or_kpts, device=device) if isinstance(images_or_kpts, list) else np.asarray(images_or_kpts)
    metrics = compute_kinematic_gait_metrics(kpts, fps)
    return embed_keypoints(kpts, device=device), metrics
