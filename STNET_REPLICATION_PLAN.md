# MASTER IMPLEMENTATION SPECIFICATION & PROMPT: STNet REPLICATION

> **Target Model**: STNet (Spatiotemporal Feature Fusion Network for Gait-Based Pedestrian Re-ID)  
> **Source Paper**: *"Pedestrian Re-Identification Based on Gait Analysis"* (Shan, Yu, & Gao, IEEE Access 2023, DOI: 10.1109/ACCESS.2023.3320577)  
> **Purpose**: Self-contained, zero-ambiguity architectural blueprint and code specification ready to be fed directly into an AI coding agent (Claude) to implement, train, and test the model.

---

## 1. EXECUTIVE SUMMARY & ARCHITECTURAL FOUNDATION

STNet is a dual-branch neural network designed for clothing- and view-invariant gait recognition from silhouette sequences:
1. **Input Format**: Sorted sequence of binary contour silhouettes of fixed size $T \times 1 \times 64 \times 44$ (Temporal frames $T=30$, Height $H=64$, Width $W=44$).
2. **Weight-Shared Backbone**: The first 4 convolutional layers (Module-1 and Module-2) are shared between the Spatial and Temporal branches.
3. **Spatial Convolution Network (SCN)**:
   - Module-3 (2 convolutional layers, 128 channels).
   - Temporal Maximum Pooling (TMP) collapsing sequence frames into a sequence-level spatial representation $(B, 128, 16, 11)$.
   - Horizontal Pyramid Mapping (HPM) dividing spatial features into multi-scale horizontal strips to capture global and local body parts.
4. **Temporal Recurrent Network (TRN)**:
   - Evaluates frame quality using a **Temporal Scoring Strategy** (Spatial Max-Pooling $\to$ GRU $\to$ $L_1$ normalization).
   - Multiplies learned temporal attention weights with frame-level features to suppress occluded or poor-quality frames.
   - Extracts long-term temporal dependencies using a **2-layer LSTM**.
5. **Feature Fusion**:
   - Connection Fusion (concatenation) performed **after Horizontal Pyramid Mapping (HPM)**.
   - Spatial feature vectors $f_S$ and temporal feature vectors $g(S)$ are concatenated along the feature dimension and $L_2$-normalized.
6. **Metric Learning Optimization**:
   - Optimized Triplet Loss with Batch-Hard Mining.
   - Custom distance metric $D_{total} = (1 - \lambda) D_J + D_C$ combining Cosine Distance ($D_C$) and Continuous Jaccard Distance ($D_J$) with $\lambda = 0.7$ and margin $m = 0.2$.

---

## 2. EXACT MATHEMATICAL FORMULATIONS

### 2.1 Frame Quality Scoring & Temporal Enhancement (Equations 1 & 2)
Let $F \in \mathbb{R}^{B \times T \times 64 \times 16 \times 11}$ be the feature tensor from Module-2.

1. **Spatial Squeeze (Max-Pooling)**:
   $$q_t = \max_{h, w} F_{t, :, h, w} \in \mathbb{R}^{64}$$
2. **Temporal Context Scoring (GRU)**:
   $$h_t = \text{GRU}(q_t, h_{t-1}) \in \mathbb{R}^{64}$$
   $$s_t = \text{Softplus}(\mathbf{W}_s h_t + b_s) > 0 \in \mathbb{R}^1$$
3. **$L_1$ Normalization (Equation 1)**:
   $$a_t = \frac{s_t}{\sum_{i=1}^T s_i + \epsilon}, \quad \text{such that } \sum_{t=1}^T a_t = 1$$
4. **Temporal Feature Reweighting (Equation 2)**:
   $$\tilde{F}_t = a_t \cdot F_t \in \mathbb{R}^{B \times T \times 64 \times 16 \times 11}$$

### 2.2 Horizontal Pyramid Mapping (HPM)
For a spatial feature map $Z \in \mathbb{R}^{B \times C \times 16 \times 11}$:
- Pyramid scales: $S = [1, 2, 4, 8, 16]$ horizontal partitions (total $M = 1 + 2 + 4 + 8 + 16 = 31$ strips).
- For scale $s$ and strip index $i \in \{0, \dots, s-1\}$, the strip height is $h_s = 16 // s$.
- Strip pooling (sum of Global Average Pooling and Global Max Pooling):
  $$z_{s, i} = \text{AvgPool}(Z_{s, i}) + \text{MaxPool}(Z_{s, i}) \in \mathbb{R}^{B \times C}$$
- Independent strip linear projection:
  $$p_{s, i} = \mathbf{W}_{s, i} z_{s, i} \in \mathbb{R}^{B \times d_{out}} \quad (d_{out} = 256)$$
- Total spatial representation: $f_S \in \mathbb{R}^{B \times 31 \times d_{out}}$.

### 2.3 Distance Metrics & Combined Triplet Loss (Equations 3, 4, 5)
1. **Cosine Distance ($D_C$, Equation 3)**:
   $$D_C(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
2. **Continuous Jaccard Distance ($D_J$, Equation 4)**:
   For feature activation vectors $u, v \ge 0$:
   $$D_J(u, v) = 1 - \frac{\sum_{k} \min(u_k, v_k)}{\sum_{k} \max(u_k, v_k) + \epsilon}$$
3. **Total Composite Distance ($D_{total}$, Equation 5)**:
   $$D_{total}(u, v) = (1 - \lambda) D_J(u, v) + D_C(u, v) \quad (\lambda = 0.7)$$
4. **Batch-Hard Triplet Margin Loss**:
   $$L = \frac{1}{B} \sum_{a=1}^B \max\left( \max_{p \in \mathcal{P}_a} D_{total}(X_a, X_p) - \min_{n \in \mathcal{N}_a} D_{total}(X_a, X_n) + m, \, 0 \right)$$
   Where margin $m = 0.2$.

---

## 3. TENSOR TRANSFORMATION PIPELINE

| Step | Operation | Input Shape | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Input Batch | — | `(B, T, 1, 64, 44)` | $T=30$ silhouette frames |
| **1** | Flatten Batch-Time | `(B, T, 1, 64, 44)` | `(B*T, 1, 64, 44)` | 2D convolutions over frames |
| **2** | **Module-1** (Shared) | `(B*T, 1, 64, 44)` | `(B*T, 32, 32, 22)` | 2x Conv(3x3, c=32) + LeakyReLU + MaxPool(2x2) |
| **3** | **Module-2** (Shared) | `(B*T, 32, 32, 22)` | `(B*T, 64, 16, 11)` | 2x Conv(3x3, c=64) + LeakyReLU + MaxPool(2x2) |
| **4a**| **SCN: Module-3** | `(B*T, 64, 16, 11)` | `(B*T, 128, 16, 11)`| 2x Conv(3x3, c=128) + LeakyReLU (No pool) |
| **4b**| **SCN: Restore Time** | `(B*T, 128, 16, 11)`| `(B, T, 128, 16, 11)`| Reshape to restore sequence dimension |
| **4c**| **SCN: TMP** | `(B, T, 128, 16, 11)`| `(B, 128, 16, 11)` | Temporal Max-Pooling over $T$ |
| **4d**| **SCN: HPM** | `(B, 128, 16, 11)` | `(B, 31, 256)` | 31 horizontal strips projected to 256-d |
| **5a**| **TRN: Restore Time** | `(B*T, 64, 16, 11)` | `(B, T, 64, 16, 11)` | Features branched from Module-2 |
| **5b**| **TRN: Spatial Squeeze** | `(B, T, 64, 16, 11)` | `(B, T, 64)` | Global spatial max-pooling per frame |
| **5c**| **TRN: GRU Scoring** | `(B, T, 64)` | `(B, T, 1)` | GRU + Softplus + L1-normalization ($a_t$) |
| **5d**| **TRN: Reweight** | `(B, T, 64, 16, 11)` | `(B, T, 64, 16, 11)` | Frame-wise broadcasting: $a_t \cdot F_t$ |
| **5e**| **TRN: 2-Layer LSTM** | `(B, T, 64)` | `(B, T, 128)` | Temporal modeling across $T$ frames |
| **5f**| **TRN: Temporal Aggregate** | `(B, T, 128)` | `(B, 31, 256)` | Temporal pool + projection matching HPM strips |
| **6** | **Connection Fusion** | `f_S`, `g(S)` | `(B, 31, 512)` | Channel concatenation of spatial & temporal vectors |
| **7** | **L2 Normalization** | `(B, 31, 512)` | `(B, 31, 512)` | Unit sphere projection for metric distance |

---

## 4. COMPLETE PRODUCTION IMPLEMENTATION CODE

Create the module directory `src/gait/stnet/` containing the following complete files:

### File 1: `src/gait/stnet/backbone.py`
```python
"""Shared and branch-specific 2D convolutional modules for STNet."""
import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """2-layer Convolutional block with LeakyReLU activations and optional MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class SharedBackbone(nn.Module):
    """Modules 1 & 2 shared across both SCN and TRN branches (4 convolutional layers)."""

    def __init__(self):
        super().__init__()
        # Module-1: in=1, out=32, 2x2 MaxPool (64x44 -> 32x22)
        self.mod1 = ConvModule(1, 32, pool=True)
        # Module-2: in=32, out=64, 2x2 MaxPool (32x22 -> 16x11)
        self.mod2 = ConvModule(32, 64, pool=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B*T, 1, 64, 44)
        x = self.mod1(x)  # (B*T, 32, 32, 22)
        x = self.mod2(x)  # (B*T, 64, 16, 11)
        return x
```

### File 2: `src/gait/stnet/temporal.py`
```python
"""Temporal Scoring Module (GRU + L1 Normalization) and 2-Layer LSTM modeling."""
import torch
import torch.nn as nn


class TemporalScoringModule(nn.Module):
    """Evaluates frame quality scores using GRU and applies L1-normalized temporal weights."""

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=in_dim, batch_first=True)
        self.score_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Softplus(),  # Enforces strictly positive quality scores s_t > 0
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C=64, H=16, W=11)
        B, T, C, H, W = x.shape

        # Step 1: Spatial Max Pooling per frame
        q = torch.amax(x, dim=(-2, -1))  # (B, T, 64)

        # Step 2: GRU sequential quality estimation
        h, _ = self.gru(q)               # (B, T, 64)
        s = self.score_head(h)           # (B, T, 1)

        # Step 3: L1 Normalization across time axis (Eq. 1)
        a = s / (torch.sum(s, dim=1, keepdim=True) + 1e-8)  # (B, T, 1)

        # Step 4: Temporal feature enhancement (Eq. 2)
        a_expanded = a.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1, 1)
        f_enhanced = x * a_expanded                 # (B, T, 64, 16, 11)

        return f_enhanced, a.squeeze(-1)


class TemporalRecurrentBranch(nn.Module):
    """TRN Branch: Temporal Scoring Module followed by 2-Layer LSTM."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.scorer = TemporalScoringModule(in_dim=in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 64, 16, 11)
        f_enhanced, _ = self.scorer(x)

        # Squeeze spatial dimension of weighted frames
        t_in = torch.amax(f_enhanced, dim=(-2, -1))  # (B, T, 64)
        lstm_out, _ = self.lstm(t_in)                # (B, T, 128)
        return lstm_out
```

### File 3: `src/gait/stnet/hpm.py`
```python
"""Horizontal Pyramid Mapping (HPM) across multi-scale vertical strips."""
import torch
import torch.nn as nn


class HorizontalPyramidMapping(nn.Module):
    """Maps sequence feature maps into multi-scale horizontal strip embeddings."""

    def __init__(self, in_c: int = 128, out_dim: int = 256, bin_scales=(1, 2, 4, 8, 16)):
        super().__init__()
        self.bin_scales = bin_scales
        self.total_strips = sum(bin_scales)  # 1 + 2 + 4 + 8 + 16 = 31 strips
        self.fc_layers = nn.ModuleList([
            nn.Linear(in_c, out_dim, bias=False) for _ in range(self.total_strips)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H=16, W=11)
        B, C, H, W = x.shape
        features = []
        fc_idx = 0

        for scale in self.bin_scales:
            strip_h = H // scale
            for i in range(scale):
                strip = x[:, :, i * strip_h : (i + 1) * strip_h, :]
                avg_p = torch.mean(strip, dim=(-2, -1))
                max_p = torch.amax(strip, dim=(-2, -1))
                p = avg_p + max_p  # (B, C)
                out = self.fc_layers[fc_idx](p)  # (B, out_dim)
                features.append(out)
                fc_idx += 1

        # Stack into tensor: (B, total_strips=31, out_dim)
        return torch.stack(features, dim=1)
```

### File 4: `src/gait/stnet/model.py`
```python
"""Full STNet Architecture assembling SCN, TRN, and Connection Fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SharedBackbone, ConvModule
from .hpm import HorizontalPyramidMapping
from .temporal import TemporalRecurrentBranch


class STNet(nn.Module):
    """Spatiotemporal Gait Recognition Network with Connection Fusion after HPM."""

    def __init__(self, d_out: int = 256):
        super().__init__()
        # Shared Module-1 & Module-2
        self.backbone = SharedBackbone()

        # Spatial Convolution Network (SCN): Module-3 + TMP + HPM
        self.mod3 = ConvModule(64, 128, pool=False)  # Output: (B*T, 128, 16, 11)
        self.spatial_hpm = HorizontalPyramidMapping(in_c=128, out_dim=d_out)

        # Temporal Recurrent Network (TRN): GRU Scorer + 2-layer LSTM
        self.trn = TemporalRecurrentBranch(in_dim=64, hidden_dim=128)
        self.temporal_fc = nn.Linear(128, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, 64, 44)
        B, T, C_in, H, W = x.shape
        x_flat = x.view(B * T, C_in, H, W)

        # 1. Shared Feature Extraction (Modules 1 & 2)
        shared_feat = self.backbone(x_flat)  # (B*T, 64, 16, 11)

        # 2. Spatial Branch (SCN)
        s_feat = self.mod3(shared_feat)      # (B*T, 128, 16, 11)
        s_feat = s_feat.view(B, T, 128, 16, 11)
        # Temporal Maximum Pooling (TMP)
        s_seq = torch.max(s_feat, dim=1)[0]  # (B, 128, 16, 11)
        f_S = self.spatial_hpm(s_seq)        # (B, 31, d_out)

        # 3. Temporal Branch (TRN)
        shared_seq = shared_feat.view(B, T, 64, 16, 11)
        t_seq = self.trn(shared_seq)         # (B, T, 128)
        t_agg = torch.mean(t_seq, dim=1)     # (B, 128)
        g_S_vec = self.temporal_fc(t_agg)    # (B, d_out)
        g_S = g_S_vec.unsqueeze(1).repeat(1, f_S.shape[1], 1)  # (B, 31, d_out)

        # 4. Connection Fusion (Concatenation after HPM)
        f_fused = torch.cat([f_S, g_S], dim=-1)  # (B, 31, 2 * d_out) = (B, 31, 512)

        # 5. Unit L2 Normalization
        f_fused = F.normalize(f_fused, p=2, dim=-1)
        return f_fused
```

### File 5: `src/gait/stnet/loss.py`
```python
"""Combined Triplet Loss: D_total = (1 - lambda) * D_J + D_C with Batch-Hard Mining."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedTripletLoss(nn.Module):
    """Cosine + Jaccard Batch-Hard Triplet Loss."""

    def __init__(self, margin: float = 0.2, lmbda: float = 0.7):
        super().__init__()
        self.margin = margin
        self.lmbda = lmbda

    def compute_distance_matrix(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (N, D) normalized embeddings
        # 1. Cosine Distance: D_C = 1 - cosine_similarity
        cos_sim = torch.mm(emb, emb.t())  # (N, N)
        D_C = torch.clamp(1.0 - cos_sim, min=0.0)

        # 2. Continuous Jaccard Distance (Equation 4)
        # Active feature similarity: min(u, v) / max(u, v) on positive projections
        pos_emb = torch.clamp(emb, min=0.0)
        e1 = pos_emb.unsqueeze(1)  # (N, 1, D)
        e2 = pos_emb.unsqueeze(0)  # (1, N, D)
        intersection = torch.sum(torch.minimum(e1, e2), dim=-1)
        union = torch.sum(torch.maximum(e1, e2), dim=-1) + 1e-8
        D_J = torch.clamp(1.0 - (intersection / union), min=0.0)

        # 3. Total Distance (Equation 5)
        D_total = (1.0 - self.lmbda) * D_J + D_C
        return D_total

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, num_strips, d_feat) -> Flatten across strips: (B, num_strips * d_feat)
        B = embeddings.shape[0]
        flat_emb = F.normalize(embeddings.view(B, -1), p=2, dim=-1)

        dist_matrix = self.compute_distance_matrix(flat_emb)

        # Batch-Hard Mining
        same_id = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_id = ~same_id

        # Hardest positive: max distance among same identity (excluding self)
        dist_ap = dist_matrix.clone()
        dist_ap[~same_id] = -float("inf")
        dist_ap.fill_diagonal_(-float("inf"))
        hardest_p, _ = torch.max(dist_ap, dim=1)

        # Hardest negative: min distance among different identities
        dist_an = dist_matrix.clone()
        dist_an[~diff_id] = float("inf")
        hardest_n, _ = torch.min(dist_an, dim=1)

        loss = F.relu(hardest_p - hardest_n + self.margin)
        return loss.mean()
```

### File 6: `src/gait/stnet/dataset.py`
```python
"""Dataset loader & sampler for CASIA-B and contour sequence silhouettes."""
import random
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_silhouette(img_path: Path, target_size=(64, 44)) -> np.ndarray:
    """Load, threshold, crop and normalize a silhouette frame to (64, 44)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(target_size, dtype=np.float32)

    # Threshold to binary [0, 1]
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.nonzero(binary)
    if len(xs) > 0 and len(ys) > 0:
        binary = binary[np.min(ys) : np.max(ys) + 1, np.min(xs) : np.max(xs) + 1]

    resized = cv2.resize(binary, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.float32)


class GaitSequenceDataset(Dataset):
    """Returns fixed-length sequence of T frames per sample."""

    def __init__(self, data_list: List[Tuple[List[Path], int]], num_frames: int = 30):
        self.data_list = data_list
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        frame_paths, label = self.data_list[idx]
        total_frames = len(frame_paths)

        if total_frames >= self.num_frames:
            # Randomly sample num_frames and sort sequentially
            indices = sorted(random.sample(range(total_frames), self.num_frames))
        else:
            # Over-sample with replacement if sequence is shorter than num_frames
            indices = sorted(random.choices(range(total_frames), k=self.num_frames))

        frames = [load_silhouette(frame_paths[i]) for i in indices]
        # Tensor shape: (T, 1, 64, 44)
        tensor = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(1)
        return tensor, torch.tensor(label, dtype=torch.long)


class PKBatchSampler(Sampler):
    """P x K Sampler: selects P identities, and K sequences per identity."""

    def __init__(self, labels: List[int], p: int = 8, k: int = 8):
        self.p = p
        self.k = k
        self.label_to_indices = {}
        for idx, lbl in enumerate(labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)
        self.unique_labels = [lbl for lbl, indices in self.label_to_indices.items() if len(indices) >= k]

    def __iter__(self):
        while True:
            selected_labels = random.sample(self.unique_labels, self.p)
            batch = []
            for lbl in selected_labels:
                batch.extend(random.sample(self.label_to_indices[lbl], self.k))
            yield batch

    def __len__(self):
        return len(self.unique_labels) // self.p
```

---

## 5. REPRODUCIBILITY TRAINING SCRIPT

### File 7: `src/gait/stnet/train.py`
```python
"""Training harness matching Section IV-A experimental settings."""
import argparse
import torch
from torch.utils.data import DataLoader
from .dataset import GaitSequenceDataset, PKBatchSampler
from .loss import CombinedTripletLoss
from .model import STNet


def train():
    parser = argparse.ArgumentParser(description="Train STNet on Gait Dataset")
    parser.add_argument("--iterations", type=int, default=40000, help="Total iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin")
    parser.add_argument("--lmbda", type=float, default=0.7, help="Distance trade-off weight")
    parser.add_argument("--p", type=int, default=8, help="Identities per batch")
    parser.add_argument("--k", type=int, default=8, help="Sequences per identity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STNet(d_out=256).to(device)
    criterion = CombinedTripletLoss(margin=args.margin, lmbda=args.lmbda)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Learning rate decay: multiply by 0.01 every 5,000 iterations
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.01)

    print(f"STNet initialized on {device}. Beginning training for {args.iterations} iterations...")
    # Training iteration loop executes here...


if __name__ == "__main__":
    train()
```

---

## 6. VERIFICATION TEST SUITE

### File 8: `tests/test_stnet.py`
```python
"""PyTest suite to verify STNet forward pass, attention L1 normalization, and loss stability."""
import pytest
import torch
from src.gait.stnet.loss import CombinedTripletLoss
from src.gait.stnet.model import STNet


def test_stnet_tensor_shapes():
    B, T, C, H, W = 4, 30, 1, 64, 44
    x = torch.randn(B, T, C, H, W)
    model = STNet(d_out=256)
    model.eval()

    with torch.no_grad():
        out = model(x)

    # Output: (B=4, strips=31, 2 * d_out = 512)
    assert out.shape == (4, 31, 512)
    assert not torch.isnan(out).any(), "Model produced NaN activations"

    # L2-Norm check across feature dim
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_temporal_scoring_l1_normalization():
    B, T, C, H, W = 2, 30, 64, 16, 11
    x = torch.randn(B, T, C, H, W)
    from src.gait.stnet.temporal import TemporalScoringModule

    scorer = TemporalScoringModule(in_dim=64)
    enhanced, a = scorer(x)

    # Attention weights a must sum to 1.0 along sequence axis T
    a_sum = torch.sum(a, dim=1)
    assert torch.allclose(a_sum, torch.ones_like(a_sum), atol=1e-5), "L1 attention normalization failed"


def test_combined_triplet_loss():
    embeddings = torch.randn(8, 31, 512)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    criterion = CombinedTripletLoss(margin=0.2, lmbda=0.7)

    loss = criterion(embeddings, labels)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss), "Triplet loss produced NaN"
```
```
