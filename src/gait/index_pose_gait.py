"""Batch Indexer for 2D Skeleton-based Pose Gait Biometrics.
Iterates over person tracks, extracts 17 COCO skeletal keypoints per frame,
runs ST-GCN to extract 256-d pose gait vectors, and indexes them into Qdrant `person_pose_gait`.
Also records biomechanical gait metrics (cadence, step length, knee flexion, symmetry) into SQLite.

Usage:
    python -m src.gait.index_pose_gait [--limit 500] [--batch 64]
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.gait.pose_gait import encode_pose_gait
from src.index import vector_db
from src.index.sql_db import connect


def process_tracks(limit: int = 500, recreate: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db = connect()

    if recreate:
        vector_db.recreate("person_pose_gait")
        print("Recreated collection 'person_pose_gait'.")

    rows = db.execute(
        "SELECT t.id, t.chunk_id, t.crops_dir, t.fps, t.path_json, t.attrs_json, c.camera, t.start_s, t.end_s "
        "FROM tracks t JOIN chunks c ON c.id=t.chunk_id "
        "WHERE t.kind='person' ORDER BY t.id LIMIT ?",
        (limit,),
    ).fetchall()

    if not rows:
        print("No person tracks found to process.")
        return

    print(f"Extracting 2D skeleton pose-gait for {len(rows)} tracks on device '{device}'...")

    ids, vecs, payloads = [], [], []
    count_indexed = 0

    for r in tqdm(rows, desc="Pose-Gait Indexing"):
        crops_dir = Path(r["crops_dir"]) if r["crops_dir"] else None
        if not crops_dir or not crops_dir.exists():
            continue

        crop_files = sorted(crops_dir.glob("*.jpg")) + sorted(crops_dir.glob("*.png"))
        if len(crop_files) < 6:
            # Need minimum frames for a gait cycle
            continue

        images = [cv2.imread(str(f)) for f in crop_files if f.exists()]
        images = [im for im in images if im is not None and im.shape[0] >= 32 and im.shape[1] >= 16]
        if len(images) < 6:
            continue

        fps = float(r["fps"]) if r["fps"] else 25.0
        emb, bio_metrics = encode_pose_gait(images, device=device, fps=fps)

        if emb is None:
            continue

        attrs = json.loads(r["attrs_json"] or "{}")
        attrs.update(bio_metrics)

        # Update SQLite with biometric gait kinematics
        db.execute("UPDATE tracks SET attrs_json=? WHERE id=?", (json.dumps(attrs), r["id"]))

        ids.append(r["id"])
        vecs.append(emb.tolist())
        payloads.append({
            "track_row": r["id"],
            "chunk_id": r["chunk_id"],
            "camera": r["camera"],
            "start_s": r["start_s"],
            "end_s": r["end_s"],
            **attrs,
        })

        if len(ids) >= 64:
            vector_db.upsert("person_pose_gait", ids, vecs, payloads)
            count_indexed += len(ids)
            ids, vecs, payloads = [], [], []

    if ids:
        vector_db.upsert("person_pose_gait", ids, vecs, payloads)
        count_indexed += len(ids)

    db.commit()
    print(f"Successfully indexed {count_indexed} tracks into 'person_pose_gait'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index 2D skeleton pose gait embeddings into Qdrant")
    parser.add_argument("--limit", type=int, default=1000, help="Max tracks to process")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    args = parser.parse_args()
    process_tracks(limit=args.limit, recreate=args.recreate)
