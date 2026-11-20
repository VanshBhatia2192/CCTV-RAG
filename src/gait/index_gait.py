"""Build `person_gait` vectors (256-d, Cosine similarity) for indexed person tracks:
Extracts 17 COCO skeletal keypoints using YOLOv8-pose, normalizes coordinates for scale
and translation invariance, and encodes temporal gait dynamics via Spatial-Temporal Graph
Convolutional Networks (ST-GCN).

Replaces legacy binary silhouette segmentation (DeepGaitV2) with 2D skeleton pose-guided
gait biometrics, ensuring full invariance to clothing swaps, jackets, bags, and illumination.

Works directly with already-indexed data (crops_dir or video chunks).
"""

import argparse
import json
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.gait.pose_gait import encode_pose_gait
from src.gait.track_crops import read_crops
from src.index import vector_db


def main():
    parser = argparse.ArgumentParser(description="Index 2D skeleton ST-GCN gait embeddings into Qdrant person_gait")
    parser.add_argument("--limit", type=int, default=0, help="Max tracks to process (0 = all)")
    parser.add_argument("--recreate", action="store_true", help="Recreate person_gait collection in Qdrant")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tracks already in person_gait")
    parser.add_argument("--part", default="0/1", help="i/N chunk splitting for parallel runs")
    parser.add_argument("--batch-size", type=int, default=32, help="Upsert batch size")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    db_path = "data/indexes/sqlite/meta.db"
    if not Path(db_path).exists():
        print(f"Error: Database file '{db_path}' not found. Please ensure data is indexed.")
        return

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    if args.recreate:
        vector_db.recreate("person_gait")
        vector_db.recreate("person_pose_gait")
        print("Recreated 'person_gait' and 'person_pose_gait' collections in Qdrant (256-d, Cosine).")

    # Fetch person tracks
    sql = (
        "SELECT t.id, t.chunk_id, t.crops_dir, t.fps, t.path_json, t.attrs_json, "
        "t.start_s, t.end_s, ch.camera, ch.path AS vpath "
        "FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id "
        "WHERE t.kind='person' ORDER BY t.id"
    )
    all_tracks = [dict(r) for r in db.execute(sql)]
    if not all_tracks:
        print("No person tracks found in database.")
        db.close()
        return

    # Filter by part i/N if specified
    pi, pn = (int(x) for x in args.part.split("/"))
    tracks = all_tracks[pi::pn]

    # Filter existing IDs if requested
    if args.skip_existing and not args.recreate:
        try:
            client = vector_db.client()
            have_ids = set()
            offset = None
            while True:
                pts, offset = client.scroll("person_gait", limit=2000, offset=offset, with_payload=False, with_vectors=False)
                have_ids.update(p.id for p in pts)
                if offset is None:
                    break
            tracks = [t for t in tracks if t["id"] not in have_ids]
            print(f"Skipping {len(have_ids)} already indexed tracks. {len(tracks)} remaining.")
        except Exception as e:
            print("Could not query existing gait IDs:", e)

    if args.limit > 0:
        tracks = tracks[:args.limit]

    print(f"Processing {len(tracks)} person tracks on device '{dev}' using 2D Skeleton ST-GCN...")

    ids, vecs, payloads = [], [], []
    count_indexed = 0
    t0 = time.time()

    for t in tqdm(tracks, desc="Pose-Gait Indexing"):
        tid = t["id"]
        fps = float(t["fps"]) if t["fps"] else 25.0
        images: List[np.ndarray] = []

        # Option A: Read directly from saved crops_dir (already extracted during track detection)
        crops_dir = Path(t["crops_dir"]) if t.get("crops_dir") else None
        if crops_dir and crops_dir.exists():
            files = sorted(crops_dir.glob("*.jpg")) + sorted(crops_dir.glob("*.png"))
            for f in files:
                im = cv2.imread(str(f))
                if im is not None and im.shape[0] >= 30 and im.shape[1] >= 12:
                    images.append(im)

        # Option B: Fallback to reading from video chunk if crops_dir is missing
        if len(images) < 5 and t.get("vpath") and Path(t["vpath"]).exists() and t.get("path_json"):
            try:
                pj = json.loads(t["path_json"])
                crops_dict = read_crops(t["vpath"], [pj], K=3, fmt="cxcywh")
                if crops_dict and 0 in crops_dict:
                    images = crops_dict[0]
            except Exception:
                pass

        if len(images) < 5:
            # Need minimum frames for gait cycle evaluation
            continue

        emb, bio_metrics = encode_pose_gait(images, device=dev, fps=fps)
        if emb is None:
            continue

        attrs = json.loads(t.get("attrs_json") or "{}")
        attrs.update(bio_metrics)

        # Update SQLite with extracted biomechanical markers (cadence, symmetry, etc.)
        db.execute(
            "UPDATE tracks SET attrs_json=? WHERE id=?",
            (json.dumps(attrs), tid),
        )

        p = {
            "track_row": tid,
            "chunk_id": t["chunk_id"],
            "camera": t["camera"],
            "start_s": t["start_s"],
            "end_s": t["end_s"],
            "motion": attrs.get("motion", "walking"),
            **bio_metrics,
        }

        ids.append(tid)
        vecs.append(emb.tolist())
        payloads.append(p)

        if len(ids) >= args.batch_size:
            vector_db.upsert("person_gait", ids, vecs, payloads)
            vector_db.upsert("person_pose_gait", ids, vecs, payloads)
            count_indexed += len(ids)
            ids, vecs, payloads = [], [], []

    if ids:
        vector_db.upsert("person_gait", ids, vecs, payloads)
        vector_db.upsert("person_pose_gait", ids, vecs, payloads)
        count_indexed += len(ids)

    db.commit()
    db.close()

    elapsed = time.time() - t0
    rate = count_indexed / max(0.1, elapsed)
    print(f"Successfully indexed {count_indexed}/{len(tracks)} tracks into 'person_gait' in {elapsed:.1f}s ({rate:.1f} tracks/s).")


if __name__ == "__main__":
    main()
