# 2D Skeleton-Guided Gait Biometrics & Cloth-Invariant Re-ID Demo

An academic presentation dashboard and interactive simulation built to demonstrate **2D Skeleton Pose-Guided Gait Biometrics** and **Clothing-Invariant Person Re-Identification** to professors, researchers, and technical stakeholders.

---

## 🚀 How to Run the Demo

From the project root:

```powershell
# Using PowerShell
.\demo_showcase\run_demo.ps1

# Or directly with uvicorn:
uvicorn demo_showcase.app:app --host 127.0.0.1 --port 8501 --reload
```

Then open your browser to **[http://localhost:8501](http://localhost:8501)**.

---

## 🎓 Key Features for Your Presentation

### 1. Live 17-Keypoint Skeletal Motion & Kinematics (Tab 1)
* **Real-time Skeleton Canvas**: Interactive 17-joint COCO anatomical skeleton animated across time. Play, pause, scrub individual frames, and switch between:
  * **Casual Walk**: Normal pace (~106 steps/min), symmetric gait.
  * **Brisk Walk**: Fast pace (~136 steps/min), extended leg stride.
  * **Asymmetrical Gait**: Pathological / injured gait (0.64 symmetry) with pronounced lateral body sway.
* **Biomechanical Metrics**: Live meters displaying step cadence (FFT), normalized stride length, knee flexion dynamics, and gait symmetry index.
* **Gait Oscilloscope Waveform**: Visualizes the sinusoidal ankle separation cycle over time.
* **ST-GCN Latent Vector**: Heatmap rendering of the 256-dimensional spatial-temporal embedding.

### 2. The Cloth-Changing Re-ID Experiment (Tab 2)
Presents the classic surveillance challenge:
* **The Problem**: A suspect is seen on Camera 1 wearing a Navy Blue Winter Coat. On Camera 2, they remove the coat and wear a Bright Red Gym Hoodie.
* **Appearance Models (CLIP)**: Fail completely (**14.8%** similarity, false rejection) and trigger a false alarm on an innocent bystander wearing a blue coat (**89.2%**).
* **Silhouette Gait (DeepGaitV2)**: Degrades to **52.4%** because the heavy coat altered the silhouette contour.
* **Our 2D Skeleton ST-GCN**: Retains **94.8% identity similarity** because bone articulation and cadence are invariant to clothing changes.

### 3. Mathematical Architecture & Graph Convolution (Tab 3)
* Formulation of degree-normalized graph Laplacian propagation:
  $$\mathbf{X}^{(l+1)} = \sigma\left(\mathbf{D}^{-\frac{1}{2}}(\mathbf{A} + \mathbf{I})\mathbf{D}^{-\frac{1}{2}} \mathbf{X}^{(l)} \mathbf{W}^{(l)}\right)$$
* Interactive $17\times 17$ adjacency matrix heatmap showing anatomical joint connectivity.
