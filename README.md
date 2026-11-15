# CCTV Search: Multi-Modal Video Intelligence & Clothing-Invariant Surveillance Retrieval

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-dc2626.svg)](https://qdrant.tech/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)

An end-to-end, scaled AI surveillance platform for natural language search across **People**, **Vehicles**, and **Incidents**, coupled with **Clothing-Invariant Cross-Camera Person Re-Identification (Re-ID)** combining fine-tuned appearance embeddings and silhouette gait biometrics.

---

## 📌 Overview

Traditional CCTV surveillance search requires security personnel to manually review hours of footage or relies on brittle keyword tagging. Furthermore, conventional visual Person Re-ID systems break down when suspects change their clothes across different cameras or days.

This platform solves these core challenges through a multi-stage architecture:
1. **Motion-Gated Video Ingestion**: Optical flow motion scoring filters out static camera footage (reducing compute by ~58%).
2. **Object Detection & Multi-Object Tracking**: YOLOv8 and ByteTrack detect and associate tracklets across time.
3. **Dense Vision-Language Understanding**: Qwen2.5-VL generates detailed natural language captions and extracts structured attributes (clothing colors, vehicle types, maneuvers, kinematics).
4. **Domain-Adapted Contrastive Search**: Fine-tuned CLIP ViT-B/32 models bridge the surveillance domain gap, lifting Text-to-Person Rank-1 retrieval from $8.25\%$ to **$50.70\%$**.
5. **Clothing-Invariant Gait Recognition**: Fine-tuned DeepGaitV2 models extract $4,096$-dimensional gait representations from YOLOv8-seg silhouettes, beating appearance models by nearly **$4\times$** on cloth-changing Re-ID benchmarks.
6. **Sub-Second Multi-Modal Search**: Reciprocal Rank Fusion (RRF) combines dense visual vectors, dense text vectors, and BM25 sparse search with kinematic/attribute boosting in **$<100\text{ ms}$**.
7. **Full-Stack Application**: FastAPI backend and single-page web application featuring HTTP 206 chunk-relative video playback and interactive cross-camera Re-ID exploration.

---

## 🏗️ Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      RAW CCTV VIDEO (400+ Videos)                      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                     [ 10-Second Non-Overlapping Chunks ]
                                     │
                     [ Motion Gate: p10/p50/p90 Filter ]
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
      [ YOLOv8 + ByteTrack ]                     [ Temporal Windows ]
         (Person & Vehicle)                          (Incidents)
               │                                           │
       ┌───────┴───────┐                                   │
       ▼               ▼                                   ▼
 [ Person Crops ] [ Vehicle Crops ]               [ Incident Windows ]
       │               │                                   │
 ┌─────┼───────────────┼───────────────────────────────────┼──────────────┐
 │     ▼               ▼                                   ▼              │
 │  Qwen2.5-VL     Qwen2.5-VL                         Qwen2.5-VL          │
 │  (Captions)     (Captions)                         (Captions)          │
 │     │               │                                   │              │
 │     ▼               ▼                                   ▼              │
 │ Fine-Tuned CLIP  Fine-Tuned CLIP                     X-CLIP            │
 │ (clip_t2p.pt)    (clip_t2v.pt)                    (Incident Emb)       │
 │     │               │                                   │              │
 │     ▼               │                                   │              │
 │ YOLOv8-Seg +        │                                   │              │
 │ DeepGaitV2 (Gait)   │                                   │              │
 └─────┼───────────────┼───────────────────────────────────┼──────────────┘
       │               │                                   │
       └───────────────┼───────────────────────────────────┘
                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         STORAGE LAYER                                  │
 │   • SQLite (WAL): Tracks, Coordinates, Attributes, Kinematics          │
 │   • Qdrant (Vector DB): person_ft, person_gait, vehicle_ft, incident   │
 └─────────────────────┬──────────────────────────────────────────────────┘
                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     HYBRID RETRIEVAL ENGINE                            │
 │   • Dense Visual + Dense Text + BM25 Lexical                           │
 │   • Weighted Reciprocal Rank Fusion (RRF)                              │
 │   • Kinematic Speed Gating & Attribute Rule Boosting                   │
 └─────────────────────┬──────────────────────────────────────────────────┘
                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      APPLICATION LAYER                                 │
 │   • FastAPI REST APIs: /api/search, /api/reid, /api/clip               │
 │   • Video Streaming: HTTP Range (206 Partial Content)                  │
 │   • Web UI: Natural Language Search & Cross-Camera Re-ID Explorer      │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark & Evaluation Results

### 1. Text-to-Person Retrieval (RSTPReid Test Set)
Evaluated across 2,000 test query descriptions against the person gallery:

| Model | Rank-1 (R@1) | Rank-5 (R@5) | Rank-10 (R@10) | mAP |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot Baseline (OpenAI CLIP ViT-B/32)** | 8.25% | 21.45% | 32.80% | 6.83% |
| **Fine-Tuned Model (`clip_t2p.pt`)** | **50.70%** | **77.00%** | **86.25%** | **40.56%** |
| **Relative Gain** | **+514.5% (6.1x)** | **+259.0% (3.6x)** | **+163.0% (2.6x)** | **+493.8% (5.9x)** |

### 2. Scaled Gallery Benchmark (13,686 Uncurated Tracks)
Evaluated on 300 natural language pseudo-queries over the full scaled 13,686-track surveillance gallery:

| Search Mode | R@1 | R@5 | R@10 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Visual Zero-Shot** | 0.67% | 3.33% | 5.00% | 0.0225 |
| **Visual Fine-Tuned (`visual_ft`)** | **1.00%** | **7.33%** | **14.33%** | **0.0535** |
| **Improvement** | **1.5x** | **2.2x** | **2.9x** | **2.4x** |

### 3. Cross-Camera Re-ID & Gait Fusion (MEVID Paired Benchmark)

#### A. Standard Cross-Camera Setup (`all`)
| Method | Rank-1 | Rank-5 | Rank-10 | mAP |
| :--- | :---: | :---: | :---: | :---: |
| Zero-Shot Appearance (`app_zs`) | 10.9% | 24.9% | 35.2% | 4.3% |
| Fine-Tuned Appearance (`app_ft`) | 44.7% | 60.8% | 69.6% | 15.9% |
| Fine-Tuned Gait Alone (`deepgaitv2_gait3d_ft`) | 22.2% | 39.9% | 48.5% | 9.2% |
| **Fused Appearance + Gait** | **46.1%** | **68.6%** | **74.1%** | **17.4%** |

#### B. Cloth-Changing Challenge (`cc` - Person changed clothes between cameras)
| Method | Rank-1 | Rank-5 | Rank-10 | mAP |
| :--- | :---: | :---: | :---: | :---: |
| Fine-Tuned Appearance (`app_ft`) | 1.5% | 6.1% | 10.7% | 2.8% |
| **Fine-Tuned Gait Alone (`deepgaitv2_gait3d_ft`)** | **5.7%** | **13.4%** | **19.5%** | **4.2%** |

> **Key Insight**: Under clothing changes, appearance features degrade to near-chance ($1.5\%$). Gait recognition maintains identity invariance, yielding **$3.8\times$ higher Rank-1 accuracy**.

### 4. End-to-End Activity Search (Multi-Modal Hybrid Fusion)
| Modality / Mode | Rank-1 | Rank-5 | Rank-10 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Person: Visual Only** | 11.11% | 22.22% | 27.78% | 0.1811 |
| **Person: Hybrid Fine-Tuned (`hybrid_ft`)** | **16.67%** | **30.56%** | **44.44%** | **0.2456** |
| **Incident: Visual Only (X-CLIP)** | 5.56% | 22.22% | 41.67% | 0.1594 |
| **Incident: Hybrid Fused** | **16.67%** | **50.00%** | **72.22%** | **0.3276** |

### 5. System Latency
- **Direct Multi-Modal Search**: **$96.3\text{ ms}$** median latency.
- **LLM Query Decomposed Search**: **$306.5\text{ ms}$** median latency.

---

## 📁 Repository Structure

```
cctv-search/
├── app/                        # FastAPI Web Serving Layer
│   ├── main.py                 # REST API & endpoints (/api/search, /api/reid, /api/clip)
│   └── static/                 # Single-page web application frontend
│       └── index.html          # Interactive search & Re-ID interface
├── configs/                    # Pipeline & model configuration YAMLs
│   ├── fusion.json             # Tuned RRF fusion weights (visual, text, lexical)
│   ├── models.yaml             # Model paths & hyperparameter configs
│   ├── paths.yaml              # Database and storage paths
│   └── pipeline.yaml           # Chunking and ingestion parameters
├── eval/                       # Evaluation & Benchmark Harnesses
│   ├── e2e/                    # End-to-end activity retrieval evaluations
│   ├── gait/                   # MEVID gait evaluation & fusion scoring
│   ├── person/                 # RSTPReid & pseudo-query evaluation scripts
│   └── latency_scaled.py       # Latency benchmarking suite
├── scripts/                    # Cluster Execution & Operational Scripts
│   ├── gpu_job.pbs             # PBS cluster batch script for GPU jobs
│   ├── gpu_train.pbs           # PBS cluster batch script for model training
│   ├── job_index_gait.sh       # Batched gait feature extraction & Qdrant indexing
│   ├── job_t2p.sh              # Text-to-Person contrastive training job
│   ├── job_t2v.sh              # Text-to-Vehicle contrastive training job
│   ├── run_demo.sh             # Launch script for Qdrant and FastAPI web server
│   ├── sync_up.sh              # Workstation-to-server sync utility
│   └── sync_down.sh            # Server-to-workstation artifact sync utility
├── src/                        # Core Python Package
│   ├── captions/               # VLM captioning & hybrid search engine
│   │   ├── caption_tracks.py   # Qwen2.5-VL tracklet caption generator
│   │   ├── caption_windows.py  # Temporal activity captioning
│   │   ├── hybrid.py           # Multi-modal RRF retrieval & attribute booster
│   │   └── text_index.py       # Dense text embeddings & BM25 sparse index
│   ├── common/                 # Shared utilities (CLIP, video IO, kinematics)
│   ├── gait/                   # Gait silhouette segmentation & DeepGaitV2 matching
│   │   ├── index_gait.py       # Parallel silhouette extraction & gait indexing
│   │   ├── og.py               # OpenGait model loader and feature extractor
│   │   ├── reid.py             # Cross-camera appearance + gait fusion logic
│   │   └── silhouettes.py      # Silhouette preprocessing & normalization
│   ├── incident/               # Temporal incident windowing & X-CLIP encoding
│   ├── index/                  # Database connectors (SQLite WAL + Qdrant)
│   ├── ingest/                 # Video chunking & motion gate filtering
│   ├── person/                 # Person detection, tracking & CLIP fine-tuning
│   ├── query/                  # Natural language query parsing & reranking
│   ├── scale/                  # Batch embedding & cluster scaling utilities
│   ├── ui/                     # Alternative Gradio interface
│   └── vehicle/                # Vehicle representation & contrastive learning
├── training/                   # Model Training Modules
│   ├── person/                 # Text-to-Person InfoNCE training
│   ├── vehicle/                # Text-to-Vehicle InfoNCE training
│   └── gait/                   # DeepGaitV2 triplet fine-tuning
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- CUDA 11.8+ or 12.x compatible GPU (for training and VLM captioning)
- Qdrant Vector Database (`bin/qdrant` or Docker container)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/cctv-search.git
cd cctv-search

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running the Interactive Demo
To run the pre-built demo locally (or over an SSH tunnel from a cluster):

```bash
# Start Qdrant and the FastAPI web service
bash scripts/run_demo.sh
```

If running on a remote server, create an SSH tunnel from your local machine:
```bash
ssh -L 8000:localhost:8000 user@server_ip
```
Then open [`http://localhost:8000`](http://localhost:8000) in your browser.

---

## 🔍 Core Workflows

### 1. Video Ingestion & Motion Gating
Split raw CCTV footage into 10-second chunks and filter out static/idle footage:
```bash
python -m src.ingest.chunk /path/to/surveillance_video.mp4
```

### 2. Multi-Target Tracking
Detect and track persons and vehicles with YOLOv8 + ByteTrack:
```bash
python -m src.person.detect_track --kind person
python -m src.person.detect_track --kind vehicle
```

### 3. VLM Captioning
Generate descriptive natural language captions using Qwen2.5-VL:
```bash
python -m src.captions.caption_tracks --kind all
```

### 4. Gait Silhouette Extraction & Indexing
Extract normalized $64\times 64$ silhouettes via YOLOv8-seg and index $4,096$-d gait vectors into Qdrant:
```bash
python -m src.gait.index_gait --part 0/1 --fmt cxcyhw --K 3
```

### 5. Fine-Tuning Vision-Language Models
Train contrastive text-to-visual models using symmetric InfoNCE loss:
```bash
# Fine-tune Text-to-Person
python -m training.person.train --epochs 8

# Fine-tune Text-to-Vehicle
python -m training.vehicle.train --epochs 8
```

---

## 🛠️ Tech Stack

- **Computer Vision**: PyTorch, Torchvision, Ultralytics YOLOv8, ByteTrack, OpenGait
- **Vision-Language Models**: OpenAI CLIP (ViT-B/32), Microsoft X-CLIP, Qwen2.5-VL-7B
- **Databases**: Qdrant (HNSW Cosine & Euclidean Indexing), SQLite (WAL Mode)
- **Search & Ranking**: Rank-Aware BM25, Reciprocal Rank Fusion (RRF)
- **API & Frontend**: FastAPI, Uvicorn, Vanilla HTML5/CSS3/ES6 JavaScript (No bulky node_modules)

---

## 📄 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

