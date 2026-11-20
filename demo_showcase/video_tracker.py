"""Video Processing, Pose Tracking, Gait Biometric Matching, and Re-Identification Alert Engine.
Takes any uploaded CCTV or surveillance video:
1. Detects and tracks all persons across frames using YOLOv8-pose.
2. Extracts 17 COCO skeletal keypoints for every tracking path.
3. Computes ST-GCN gait biometrics (256-d) and kinematic parameters (cadence, stride, symmetry).
4. Matches gait signatures across time: when a person disappears and reappears, it triggers
   a real-time 'PERSON IDENTIFIED' alert with timestamp, confidence, and side-by-side thumbnails.
"""

import base64
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.gait.pose_gait import (
    compute_kinematic_gait_metrics,
    encode_pose_gait,
    get_pose_detector,
)


def iou(boxA, boxB):
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(dot / norm)


def encode_image_base64(img_bgr: np.ndarray, max_size: int = 180) -> str:
    """Resize and encode image as base64 data URI."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = max_size / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def process_video_and_detect_gait_alerts(
    video_path: str,
    output_dir: Path,
    sample_stride: Optional[int] = None,
    match_threshold: float = 0.82,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """Process video file, track people, extract gait signatures, and trigger re-appearance alerts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    # Adapt sampling stride if not explicitly specified so long videos finish in ~15-25 seconds on CPU
    if sample_stride is None or sample_stride <= 0:
        target_frames = 200
        if total_frames > target_frames:
            sample_stride = max(2, int(total_frames / target_frames))
        else:
            sample_stride = 2

    if progress_callback:
        progress_callback(5, f"Initializing YOLOv8-pose tracker (stride={sample_stride}, total={total_frames} frames)...")

    detector = get_pose_detector()

    # Track management
    active_tracks = []   # list of currently open tracks
    completed_tracks = []
    next_track_id = 1
    max_lost_frames = int(fps * 2.0)  # track lost after 2.0 seconds

    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        processed_count += 1
        current_time_s = frame_idx / fps

        if progress_callback and processed_count % 10 == 0:
            pct = int(min(88, 5 + (frame_idx / max(1, total_frames)) * 83))
            progress_callback(pct, f"Analyzing frame {frame_idx}/{total_frames} ({len(active_tracks)} active subjects)...")

        # Run YOLO-pose at imgsz=256 for fast CPU inference
        res = detector.predict(frame, imgsz=256, conf=0.25, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0 and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for box, kpts in zip(boxes, kpts_data):
                w_box = box[2] - box[0]
                h_box = box[3] - box[1]
                # Filter noise
                if h_box >= 25 and w_box >= 10:
                    x1 = max(0, int(box[0]))
                    y1 = max(0, int(box[1]))
                    x2 = min(width, int(box[2]))
                    y2 = min(height, int(box[3]))
                    crop_bgr = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                    detections.append({
                        "box": [float(x) for x in box],
                        "kpts": kpts,
                        "frame": frame_idx,
                        "time_s": current_time_s,
                        "crop_bgr": crop_bgr,
                    })

        # Associate detections to active tracks via IoU
        matched_det_indices = set()
        for tr in active_tracks:
            best_iou = 0.0
            best_det_idx = -1
            last_box = tr["boxes"][-1]
            for di, det in enumerate(detections):
                if di in matched_det_indices:
                    continue
                score = iou(last_box, det["box"])
                if score > best_iou:
                    best_iou = score
                    best_det_idx = di

            if best_iou >= 0.20 and best_det_idx >= 0:
                det = detections[best_det_idx]
                tr["boxes"].append(det["box"])
                tr["kpts"].append(det["kpts"])
                tr["times"].append(det["time_s"])
                tr["frames"].append(det["frame"])
                tr["lost"] = 0
                # Update thumbnail to a clear mid-track crop
                if det["crop_bgr"] is not None and (tr["thumb"] is None or len(tr["boxes"]) == 4):
                    tr["thumb"] = encode_image_base64(det["crop_bgr"])
                matched_det_indices.add(best_det_idx)
            else:
                tr["lost"] += sample_stride

        # Finish tracks that are lost
        surviving_tracks = []
        for tr in active_tracks:
            if tr["lost"] >= max_lost_frames:
                completed_tracks.append(tr)
            else:
                surviving_tracks.append(tr)
        active_tracks = surviving_tracks

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                initial_thumb = encode_image_base64(det["crop_bgr"]) if det["crop_bgr"] is not None else None
                active_tracks.append({
                    "id": next_track_id,
                    "boxes": [det["box"]],
                    "kpts": [det["kpts"]],
                    "times": [det["time_s"]],
                    "frames": [det["frame"]],
                    "thumb": initial_thumb,
                    "lost": 0,
                })
                next_track_id += 1

        frame_idx += 1

    cap.release()
    completed_tracks.extend(active_tracks)

    if progress_callback:
        progress_callback(90, f"Evaluating {len(completed_tracks)} tracks with ST-GCN gait biometrics...")

    # Filter tracks with sufficient keypoints for gait analysis (>= 5 keypoint samples)
    valid_tracks = [t for t in completed_tracks if len(t["kpts"]) >= 5]

    # Process gait biometrics & ST-GCN embeddings for valid tracks
    effective_fps = max(1.0, fps / sample_stride)
    track_profiles = []
    for tr in valid_tracks:
        kpts_arr = np.array(tr["kpts"], dtype=np.float32)  # (T, 17, 3)
        metrics = compute_kinematic_gait_metrics(kpts_arr, fps=effective_fps)
        emb, _ = encode_pose_gait(kpts_arr, device="cpu", fps=effective_fps)

        if emb is not None:
            track_profiles.append({
                "track_id": tr["id"],
                "start_s": round(tr["times"][0], 1),
                "end_s": round(tr["times"][-1], 1),
                "duration_s": round(tr["times"][-1] - tr["times"][0], 1),
                "num_frames": len(tr["kpts"]),
                "embedding": emb,
                "metrics": metrics,
                "thumb": tr.get("thumb"),
            })

    # Sort tracks chronologically
    track_profiles.sort(key=lambda x: x["start_s"])

    # Cluster identities and trigger Re-Appearance Alerts
    unique_persons: List[Dict] = []
    alerts: List[Dict] = []
    alert_counter = 1

    for tp in track_profiles:
        best_match_person = None
        best_sim = 0.0

        for p in unique_persons:
            # Compare current track against the person's average gait embedding
            sim = cos_sim(tp["embedding"], p["mean_embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match_person = p

        # Check if matched to an existing person
        if best_sim >= match_threshold and best_match_person is not None:
            pid = best_match_person["person_id"]
            last_appearance = best_match_person["occurrences"][-1]
            time_gap = tp["start_s"] - last_appearance["end_s"]

            # Add occurrence
            best_match_person["occurrences"].append({
                "track_id": tp["track_id"],
                "start_s": tp["start_s"],
                "end_s": tp["end_s"],
                "thumb": tp["thumb"],
                "metrics": tp["metrics"],
            })
            # Update mean gait vector
            all_embs = [best_match_person["initial_embedding"], tp["embedding"]]
            best_match_person["mean_embedding"] = np.mean(all_embs, axis=0)

            # Trigger Alert if re-appeared after a gap (disappeared and came back)
            if time_gap >= 1.5:
                alerts.append({
                    "alert_id": f"ALERT-{alert_counter:02d}",
                    "person_id": pid,
                    "title": f"🚨 {pid} Re-Identified by Gait",
                    "reappeared_time_s": tp["start_s"],
                    "first_seen_time_s": last_appearance["start_s"],
                    "confidence_pct": round(best_sim * 100.0, 1),
                    "cadence_bpm": tp["metrics"]["cadence_bpm"],
                    "gait_symmetry": tp["metrics"]["gait_symmetry"],
                    "reappear_thumb": tp["thumb"],
                    "initial_thumb": best_match_person["initial_thumb"],
                    "message": (
                        f"{pid} was first observed at {last_appearance['start_s']:.1f}s and has "
                        f"RE-APPEARED at {tp['start_s']:.1f}s with {best_sim * 100:.1f}% gait biometric match."
                    ),
                })
                alert_counter += 1
        else:
            # New unique person identified
            new_pid = f"Person #{len(unique_persons) + 1}"
            unique_persons.append({
                "person_id": new_pid,
                "initial_embedding": tp["embedding"],
                "mean_embedding": tp["embedding"],
                "initial_thumb": tp["thumb"],
                "metrics": tp["metrics"],
                "occurrences": [{
                    "track_id": tp["track_id"],
                    "start_s": tp["start_s"],
                    "end_s": tp["end_s"],
                    "thumb": tp["thumb"],
                    "metrics": tp["metrics"],
                }],
            })

    # Prepare response payload
    serializable_persons = []
    for p in unique_persons:
        serializable_persons.append({
            "person_id": p["person_id"],
            "initial_thumb": p["initial_thumb"],
            "metrics": p["metrics"],
            "num_occurrences": len(p["occurrences"]),
            "occurrences": [
                {
                    "track_id": occ["track_id"],
                    "start_s": occ["start_s"],
                    "end_s": occ["end_s"],
                    "thumb": occ["thumb"],
                    "metrics": occ["metrics"],
                }
                for occ in p["occurrences"]
            ],
        })

    if progress_callback:
        progress_callback(100, "Processing complete")

    return {
        "video_name": Path(video_path).name,
        "duration_s": round(duration_s, 1),
        "total_frames": total_frames,
        "fps": round(fps, 1),
        "num_unique_persons": len(unique_persons),
        "num_tracks_analyzed": len(track_profiles),
        "alerts_count": len(alerts),
        "alerts": alerts,
        "persons": serializable_persons,
    }
