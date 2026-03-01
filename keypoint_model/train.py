"""
Training script for the vial keypoint heatmap model.

Usage:
    python -m keypoint_model.train \
        --annotations assets/testing_images/vial_keypoints.json \
        --images assets/testing_images \
        --splits assets/testing_images/splits.json \
        --epochs 150 --lr 1e-3 --batch-size 4 \
        --backbone resnet18 --save-dir checkpoints
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import VialKeypointDataset
from .model import VialKeypointNet
from .utils import find_peaks


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_detection_metrics(pred_heatmap: np.ndarray,
                              gt_keypoints: list[list[float]],
                              stride: int,
                              threshold: float = 0.3,
                              dist_thresh: float = 4.0):
    """
    Compute per-image TP, FP, FN and mean distance error
    at a given distance threshold (in heatmap-pixel units).
    """
    peaks = find_peaks(pred_heatmap, threshold=threshold, nms_radius=3)

    gt = np.array(gt_keypoints, dtype=np.float32)
    gt_hm = gt / stride

    if len(peaks) == 0 and len(gt_hm) == 0:
        return 1.0, 1.0, 0, 0.0
    if len(peaks) == 0:
        return 0.0, 0.0, 0, float("inf")
    if len(gt_hm) == 0:
        return 0.0, 1.0, len(peaks), 0.0

    pred_xy = np.array([[p[0], p[1]] for p in peaks], dtype=np.float32)

    matched_gt = set()
    matched_dists = []
    for pi, pxy in enumerate(pred_xy):
        dists = np.linalg.norm(gt_hm - pxy, axis=1)
        best_gi = int(np.argmin(dists))
        if dists[best_gi] < dist_thresh and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            matched_dists.append(float(dists[best_gi]))

    tp = len(matched_gt)
    precision = tp / len(peaks) if len(peaks) > 0 else 0.0
    recall = tp / len(gt_hm) if len(gt_hm) > 0 else 0.0
    mean_dist = float(np.mean(matched_dists)) if matched_dists else float("inf")
    return precision, recall, len(peaks), mean_dist


def _unpack_meta_keypoints(meta):
    """Robustly extract GT keypoints from DataLoader-collated meta dict (bs=1)."""
    raw = meta["keypoints"]
    if isinstance(raw, torch.Tensor):
        return raw[0].numpy().tolist()
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, list) and len(item) == 2:
                x_val = item[0].item() if isinstance(item[0], torch.Tensor) else float(item[0])
                y_val = item[1].item() if isinstance(item[1], torch.Tensor) else float(item[1])
                result.append([x_val, y_val])
            elif isinstance(item, torch.Tensor):
                result.append(item[0].tolist() if item.dim() > 0 else [item.item()])
        return result
    return raw


class Logger:
    """Tee stdout to both console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def train(args):
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.txt")
    logger = Logger(log_path)
    sys.stdout = logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")
    print(f"[TRAIN] Config: backbone={args.backbone}, input={args.input_h}x{args.input_w}, "
          f"sigma={args.sigma}, lr={args.lr}, bs={args.batch_size}, epochs={args.epochs}")

    with open(args.splits) as f:
        splits = json.load(f)
    train_files = splits["train"]
    test_files = splits["test"]

    random.shuffle(train_files)
    n_val = max(1, int(len(train_files) * 0.10))
    val_files = train_files[:n_val]
    train_files = train_files[n_val:]

    print(f"[TRAIN] Train: {len(train_files)}, Val: {len(val_files)}, Test (held out): {len(test_files)}")

    input_size = (args.input_h, args.input_w)
    train_ds = VialKeypointDataset(
        args.annotations, args.images,
        input_size=input_size, heatmap_stride=4,
        sigma=args.sigma, augment=True, file_list=train_files
    )
    val_ds = VialKeypointDataset(
        args.annotations, args.images,
        input_size=input_size, heatmap_stride=4,
        sigma=args.sigma, augment=False, file_list=val_files
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=0)

    model_cfg = ModelConfig(backbone=args.backbone, pretrained=True)
    model = VialKeypointNet(model_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {args.backbone} | total params: {total_params:,} | trainable: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_steps), gamma=args.lr_factor
    )

    best_val_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_precision": [], "val_recall": [], "val_mean_dist": [], "lr": []}

    total_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for images, targets, _meta in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_prec, val_rec, val_dist = 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, targets, meta in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(images)
                val_loss += criterion(preds, targets).item()

                pred_np = torch.sigmoid(preds[0, 0]).cpu().numpy()
                gt_kps = _unpack_meta_keypoints(meta)
                p, r, _, d = compute_detection_metrics(pred_np, gt_kps, stride=4)
                val_prec += p
                val_rec += r
                val_dist += d if d != float("inf") else 0.0

        n_val_batches = max(len(val_loader), 1)
        avg_val_loss = val_loss / n_val_batches
        avg_prec = val_prec / n_val_batches
        avg_rec = val_rec / n_val_batches
        avg_dist = val_dist / n_val_batches

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_precision"].append(avg_prec)
        history["val_recall"].append(avg_rec)
        history["val_mean_dist"].append(avg_dist)
        history["lr"].append(lr_now)

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={avg_train_loss:.6f} | "
                  f"val_loss={avg_val_loss:.6f} | "
                  f"P={avg_prec:.3f} R={avg_rec:.3f} dist={avg_dist:.2f} | "
                  f"lr={lr_now:.1e} | {elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_precision": avg_prec,
                "val_recall": avg_rec,
                "config": {
                    "backbone": args.backbone,
                    "input_size": input_size,
                    "sigma": args.sigma,
                },
            }, ckpt_path)
            if epoch <= 5 or epoch % 10 == 0:
                print(f"  -> saved best model (val_loss={avg_val_loss:.6f})")

        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "config": {
                    "backbone": args.backbone,
                    "input_size": input_size,
                    "sigma": args.sigma,
                },
            }, ckpt_path)

    total_time = time.time() - total_t0
    print(f"\n[TRAIN] Done in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"[TRAIN] Best val loss: {best_val_loss:.6f}")

    # Save last checkpoint
    ckpt_path = os.path.join(args.save_dir, "last_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "config": {
            "backbone": args.backbone,
            "input_size": input_size,
            "sigma": args.sigma,
        },
    }, ckpt_path)

    # Save training history
    hist_path = os.path.join(args.save_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[TRAIN] History saved -> {hist_path}")

    sys.stdout = logger.terminal
    logger.close()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train vial keypoint heatmap model")
    p.add_argument("--annotations", type=str, required=True)
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--splits", type=str, required=True,
                   help="Path to splits.json with train/test lists")
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-steps", type=int, nargs="+", default=[60, 100, 130])
    p.add_argument("--lr-factor", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--input-h", type=int, default=384)
    p.add_argument("--input-w", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
