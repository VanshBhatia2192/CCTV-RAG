#!/usr/bin/env python3
"""Unified Dataset Ingestion Pipeline for CCTV-Search.
Connects a new camera feed or dataset, chunks videos, extracts tracks,
computes multimodal embeddings, and indexes them with a dataset tag.

Usage:
  python scripts/ingest_dataset.py --src /path/to/videos --dataset cityflow --kind vehicle
  python scripts/ingest_dataset.py --src /path/to/cctv --dataset campus --camera-prefix campus.c01 --kind all
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.index.sql_db import connect


def run_ingest(src_path: str, dataset_name: str, camera_prefix: str = None, kind: str = "all", limit: int = 1000):
    t0 = time.time()
    db = connect()

    print(f"============================================================")
    print(f"🚀 CCTV-RAG Dataset Ingestion Pipeline")
    print(f"  Source:         {src_path}")
    print(f"  Dataset Tag:    {dataset_name}")
    print(f"  Camera Prefix:  {camera_prefix or '(auto-inferred from folder/filename)'}")
    print(f"  Target Classes: {kind}")
    print(f"============================================================")

    src = Path(src_path)
    if not src.exists():
        print(f"❌ Error: Source path does not exist: {src}")
        sys.exit(1)

    # Step 1: Chunking
    print("\n--- [Step 1/3] Video Chunking & Motion Gating ---")
    from src.ingest import chunk
    chunk.main(str(src), camera=camera_prefix, dataset=dataset_name)

    # Check how many chunks exist for this dataset
    n_chunks = db.execute("SELECT count(*) FROM chunks WHERE dataset=?", (dataset_name,)).fetchone()[0]
    print(f"✓ Total chunks indexed for dataset '{dataset_name}': {n_chunks}")

    kinds = ["person", "vehicle"] if kind == "all" else [kind]

    # Step 2: Object Tracking (YOLOv8 + ByteTrack)
    print("\n--- [Step 2/3] Detection & Multi-Object Tracking ---")
    from src.person import detect_track
    for k in kinds:
        print(f"\nTracking {k}s on unprocessed chunks...")
        detect_track.main(kind=k, limit=limit)

    # Step 3: Visual & Attribute Embeddings
    print("\n--- [Step 3/3] CLIP Vector Embeddings & Zero-Shot Attributes ---")
    from src.person import embed
    for k in kinds:
        print(f"\nEmbedding {k}s into vector index...")
        embed.main(kind=k, limit=limit)

    # Step 4: Fine-Tuned Tower Embeddings (if checkpoints exist)
    if "vehicle" in kinds and Path("models/finetuned/vehicle/clip_vehicle.pt").exists():
        print("\nUpdating fine-tuned vehicle visual embeddings (vehicle_ft)...")
        try:
            from src.vehicle import embed_ft
            embed_ft.main()
        except Exception as e:
            print("Note: fine-tuned vehicle embedding skipped:", e)

    elapsed = round(time.time() - t0, 1)
    
    # Final Telemetry Report
    cur_tracks = dict(db.execute(
        "SELECT t.kind, count(*) FROM tracks t JOIN chunks c ON c.id=t.chunk_id WHERE c.dataset=? GROUP BY t.kind",
        (dataset_name,)
    ).fetchall())
    cams = [r[0] for r in db.execute(
        "SELECT distinct camera FROM chunks WHERE dataset=? AND camera IS NOT NULL", (dataset_name,)
    ).fetchall()]

    print(f"\n============================================================")
    print(f"✅ Ingestion Complete in {elapsed}s!")
    print(f"  Dataset: '{dataset_name}'")
    print(f"  Cameras: {len(cams)} ({', '.join(cams[:6])}{'...' if len(cams) > 6 else ''})")
    print(f"  Tracks:  {cur_tracks}")
    print(f"  Ready to search! Select '{dataset_name.upper()}' in the web UI.")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest video files or camera folders into CCTV-RAG index")
    parser.add_argument("--src", required=True, help="Path to video file or folder containing videos")
    parser.add_argument("--dataset", required=True, help="Dataset identifier tag (e.g. cityflow, campus, parking)")
    parser.add_argument("--camera-prefix", default=None, help="Explicit camera name or prefix")
    parser.add_argument("--kind", choices=["person", "vehicle", "all"], default="all", help="Classes to detect and track")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of chunks to track in this pass")
    args = parser.parse_args()

    run_ingest(args.src, args.dataset, args.camera_prefix, args.kind, args.limit)

