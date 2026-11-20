"""Fine-tune CLIP for Multi-Camera Vehicle Re-Identification on the AI City / CityFlow dataset.
Combines:
1. Cross-Camera Identity Contrastive Loss (Image-to-Image): pulls different views of the same vehicle together.
2. Cross-Modal InfoNCE Loss (Image-to-Text): preserves text-query search alignment.
Evaluates multi-camera Rank-1, Rank-5, Rank-10, and mAP on held-out validation scene.
"""

import argparse
import glob
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.person import t2p


RX = re.compile(r"^\./?(?P<split>[^/]+)/(?P<scene>S\d+)/(?P<cam>c\d+)/img1/(?P<f>\d+)\.jpg$")


def build_identity_mapping(manifest_path, gt_glob, tracks_json_path):
    """Map every track in manifest.json to official multi-camera ground truth vehicle identity."""
    gt_map = {}
    for path in glob.glob(gt_glob):
        parts = path.split(os.sep)
        # expected: .../AICity22_Track1_MTMC_Tracking/<split>/<scene>/<cam>/gt/gt.txt
        scene, cam = parts[-4], parts[-3]
        with open(path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 6:
                    frame, vid, x, y = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                    gt_map[(scene, cam, frame, x, y)] = f"{scene}_{vid}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(tracks_json_path) as f:
        orig = json.load(f)

    track_to_vid = {}
    vid_to_tracks = defaultdict(list)

    for k in manifest:
        if k in orig and orig[k]["frames"]:
            mat = RX.match(orig[k]["frames"][0])
            if mat:
                sc, cam, fn = mat["scene"], mat["cam"], int(mat["f"])
                bx = orig[k]["boxes"][0]
                x, y = int(bx[0]), int(bx[1])
                vid = gt_map.get((sc, cam, fn, x, y))
                if vid:
                    track_to_vid[k] = vid
                    vid_to_tracks[vid].append(k)

    return manifest, track_to_vid, vid_to_tracks


def evaluate_reid(model, imgs, query_keys, gallery_keys, manifest, track_to_vid, dev):
    """Evaluate cross-camera vehicle Re-ID: CMC (Rank-1, 5, 10) and mAP."""
    model.eval()

    def get_feats(keys):
        feats = []
        for k in keys:
            arr = imgs[k]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                x = (torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255 - t2p.MEAN.to(dev)) / t2p.STD.to(dev)
                f = t2p.img_feat(model, x)
                f_norm = F.normalize(f.float(), dim=-1).mean(0)
                f_norm = F.normalize(f_norm, dim=0)
                feats.append(f_norm.cpu())
        return torch.stack(feats)

    q_feats = get_feats(query_keys)
    g_feats = get_feats(gallery_keys)

    sims = (q_feats @ g_feats.T).numpy()

    q_vids = [track_to_vid[k] for k in query_keys]
    q_cams = [manifest[k]["cam"] for k in query_keys]
    g_vids = [track_to_vid[k] for k in gallery_keys]
    g_cams = [manifest[k]["cam"] for k in gallery_keys]

    r1, r5, r10, aps = [], [], [], []

    for qi, q_vid in enumerate(q_vids):
        q_cam = q_cams[qi]
        row_sim = sims[qi].copy()

        # Cross-camera Re-ID protocol: mask out gallery items from the SAME camera as query
        valid_mask = np.array([gc != q_cam for gc in g_cams])
        if not valid_mask.any():
            continue

        pos_mask = np.array([gv == q_vid and gc != q_cam for gv, gc in zip(g_vids, g_cams)])
        if not pos_mask.any():
            continue

        row_sim[~valid_mask] = -1e9
        order = np.argsort(-row_sim)
        matches = pos_mask[order]

        # CMC ranks
        first_pos = np.where(matches)[0]
        if len(first_pos) > 0:
            rank = first_pos[0]
            r1.append(1.0 if rank < 1 else 0.0)
            r5.append(1.0 if rank < 5 else 0.0)
            r10.append(1.0 if rank < 10 else 0.0)

            # Average Precision (AP)
            n_pos = pos_mask.sum()
            cum_pos = np.cumsum(matches)
            precisions = cum_pos[matches] / (np.where(matches)[0] + 1)
            aps.append(precisions.sum() / n_pos)

    model.train()

    return {
        "Rank1": round(float(np.mean(r1) * 100), 2) if r1 else 0.0,
        "Rank5": round(float(np.mean(r5) * 100), 2) if r5 else 0.0,
        "Rank10": round(float(np.mean(r10) * 100), 2) if r10 else 0.0,
        "mAP": round(float(np.mean(aps) * 100), 2) if aps else 0.0,
        "n_queries": len(r1),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Vehicle Re-ID Fine-Tuning on AI City")
    parser.add_argument("--crops", default="data/processed/cityflow_crops")
    parser.add_argument("--gt-glob", default="data/raw/cityflow_nl/AICity22_Track1_MTMC_Tracking/*/*/*/gt/gt.txt")
    parser.add_argument("--tracks", default="data/raw/cityflow_nl/train-tracks.json")
    parser.add_argument("--val-scene", default="S02", help="Held-out scene for multi-camera Re-ID evaluation")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-i2i", type=float, default=1.0, help="Weight on Image-to-Image Re-ID loss")
    parser.add_argument("--weight-i2t", type=float, default=0.5, help="Weight on Image-to-Text loss")
    parser.add_argument("--out", default="models/finetuned/vehicle/clip_vehicle.pt")
    parser.add_argument("--report", default="outputs/reports/vehicle_reid_ft.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Multi-Camera Vehicle Re-ID Fine-Tuning | Device: {dev} ===", flush=True)

    manifest, track_to_vid, vid_to_tracks = build_identity_mapping(
        os.path.join(args.crops, "manifest.json"), args.gt_glob, args.tracks
    )

    all_vids = sorted(vid_to_tracks.keys())
    train_vids = [v for v in all_vids if not v.startswith(args.val_scene)]
    val_vids = [v for v in all_vids if v.startswith(args.val_scene)]

    # Collect tracks
    train_keys = [k for v in train_vids for k in vid_to_tracks[v]]
    val_keys = [k for v in val_vids for k in vid_to_tracks[v]]

    # For validation: split S02 cameras into query and gallery cameras
    val_cams = sorted({manifest[k]["cam"] for k in val_keys})
    mid = len(val_cams) // 2
    query_cams = set(val_cams[:mid])
    gallery_cams = set(val_cams[mid:])

    query_keys = [k for k in val_keys if manifest[k]["cam"] in query_cams]
    gallery_keys = [k for k in val_keys if manifest[k]["cam"] in gallery_cams]

    print(f"Dataset summary:", flush=True)
    print(f"  Total matched tracks: {len(track_to_vid)} across {len(all_vids)} vehicle identities", flush=True)
    print(f"  Training: {len(train_keys)} tracks across {len(train_vids)} identities (scenes != {args.val_scene})", flush=True)
    print(f"  Validation Re-ID: {len(query_keys)} queries ({query_cams}) vs {len(gallery_keys)} gallery ({gallery_cams}) in {args.val_scene}", flush=True)

    # Preload crop images
    def load_crops(k):
        return np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)) for p in manifest[k]["crops"]])

    all_keys = sorted(list(set(train_keys + val_keys)))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        imgs = dict(zip(all_keys, ex.map(load_crops, all_keys)))
    print(f"Loaded {sum(len(v) for v in imgs.values())} crops in {time.time() - t0:.1f}s", flush=True)

    # Load baseline CLIP model
    model, tok = t2p.load(None, dev)
    MEAN, STD = t2p.MEAN.to(dev), t2p.STD.to(dev)

    def prep_batch(arr):
        return (torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD

    # Evaluate zero-shot baseline
    print("\n--- Evaluating Zero-Shot Baseline ---", flush=True)
    zs_metrics = evaluate_reid(model, imgs, query_keys, gallery_keys, manifest, track_to_vid, dev)
    print(f"Zero-Shot Baseline on {args.val_scene}: Rank-1: {zs_metrics['Rank1']}% | Rank-5: {zs_metrics['Rank5']}% | Rank-10: {zs_metrics['Rank10']}% | mAP: {zs_metrics['mAP']}%", flush=True)

    # Optimizer & LR Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2, betas=(0.9, 0.98), eps=1e-6)
    steps_per_epoch = max(1, len(train_vids) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * s / total_steps))
    )

    best_score = zs_metrics["mAP"] + zs_metrics["Rank1"]
    best_state = None
    history = [dict(epoch=0, **zs_metrics)]

    print(f"\n--- Starting Multi-Camera Re-ID Fine-Tuning ({args.epochs} epochs) ---", flush=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_vids))
        total_loss, total_i2i, total_i2t = 0.0, 0.0, 0.0

        for b in range(steps_per_epoch):
            batch_vids = [train_vids[i] for i in perm[b * args.batch : (b + 1) * args.batch]]
            if len(batch_vids) < 4:
                continue

            crops_a, crops_b, texts = [], [], []

            for vid in batch_vids:
                trk_list = vid_to_tracks[vid]
                # Pick track 1
                t1 = np.random.choice(trk_list)
                crops1 = imgs[t1]
                idx1 = np.random.randint(len(crops1))
                crops_a.append(crops1[idx1])

                # Pick track 2 (prefer different track/camera if available)
                if len(trk_list) > 1:
                    t2 = np.random.choice([x for x in trk_list if x != t1])
                else:
                    t2 = t1
                crops2 = imgs[t2]
                idx2 = np.random.randint(len(crops2))
                if t2 == t1 and len(crops1) > 1 and idx2 == idx1:
                    idx2 = (idx1 + 1) % len(crops1)
                crops_b.append(crops2[idx2])

                # Natural language description
                desc = np.random.choice(manifest[t1]["nl"]) if manifest[t1]["nl"] else "a vehicle"
                texts.append(desc)

            xa = prep_batch(np.stack(crops_a))
            xb = prep_batch(np.stack(crops_b))

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                fa = t2p.img_feat(model, xa)
                fb = t2p.img_feat(model, xb)
                ft = t2p.txt_feat(model, tok, texts, dev)

            fa = F.normalize(fa.float(), dim=-1)
            fb = F.normalize(fb.float(), dim=-1)
            ft = F.normalize(ft.float(), dim=-1)

            scale = model.logit_scale.exp().clamp(max=100)
            labels = torch.arange(len(batch_vids), device=dev)

            # 1. Image-to-Image Re-ID loss
            sim_i2i = scale * (fa @ fb.T)
            loss_i2i = (F.cross_entropy(sim_i2i, labels) + F.cross_entropy(sim_i2i.T, labels)) / 2

            # 2. Image-to-Text loss
            sim_i2t = scale * (fa @ ft.T)
            loss_i2t = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_i2t.T, labels)) / 2

            # Combined Multi-Camera Re-ID objective
            loss = args.weight_i2i * loss_i2i + args.weight_i2t * loss_i2t

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            total_loss += loss.item()
            total_i2i += loss_i2i.item()
            total_i2t += loss_i2t.item()

        # Validation at epoch end
        cur_loss = total_loss / steps_per_epoch
        cur_i2i = total_i2i / steps_per_epoch
        cur_i2t = total_i2t / steps_per_epoch

        val_metrics = evaluate_reid(model, imgs, query_keys, gallery_keys, manifest, track_to_vid, dev)
        history.append(dict(epoch=ep, loss=round(cur_loss, 4), loss_i2i=round(cur_i2i, 4), loss_i2t=round(cur_i2t, 4), **val_metrics))

        print(
            f"Epoch {ep:02d}/{args.epochs:02d} | Loss: {cur_loss:.4f} (i2i: {cur_i2i:.4f}, i2t: {cur_i2t:.4f}) | "
            f"Val Re-ID: Rank-1: {val_metrics['Rank1']}% | Rank-5: {val_metrics['Rank5']}% | mAP: {val_metrics['mAP']}%",
            flush=True,
        )

        val_score = val_metrics["mAP"] + val_metrics["Rank1"]
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> New best model checkpoint (mAP={val_metrics['mAP']}%, R1={val_metrics['Rank1']}%)", flush=True)

    # Save best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print("\nLoaded best model state from training.", flush=True)

    final_metrics = evaluate_reid(model, imgs, query_keys, gallery_keys, manifest, track_to_vid, dev)
    print("\n=== FINAL EVALUATION ===", flush=True)
    print(f"Zero-Shot:   Rank-1: {zs_metrics['Rank1']}% | Rank-5: {zs_metrics['Rank5']}% | mAP: {zs_metrics['mAP']}%", flush=True)
    print(f"Fine-Tuned:  Rank-1: {final_metrics['Rank1']}% | Rank-5: {final_metrics['Rank5']}% | mAP: {final_metrics['mAP']}%", flush=True)
    gain_r1 = final_metrics['Rank1'] - zs_metrics['Rank1']
    gain_map = final_metrics['mAP'] - zs_metrics['mAP']
    print(f"Gain:        ΔRank-1: +{gain_r1:.2f}% | ΔmAP: +{gain_map:.2f}%", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), args.out)
    print(f"Saved fine-tuned vehicle Re-ID weights to {args.out}", flush=True)

    report = {
        "args": vars(args),
        "zero_shot": zs_metrics,
        "fine_tuned": final_metrics,
        "delta": {"Rank1": gain_r1, "mAP": gain_map},
        "history": history,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {args.report}", flush=True)


if __name__ == "__main__":
    main()

