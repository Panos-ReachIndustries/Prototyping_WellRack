"""
Training script for the vial keypoint heatmap model.

Usage:
    python -m keypoint_model.train \
        --annotations assets/testing_images/vial_keypoints.json \
        --images assets/testing_images \
        --epochs 200 --lr 1e-3 --batch-size 4 \
        --backbone resnet18 --save-dir checkpoints
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
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
                              dist_thresh: float = 4.0):
    """
    Compute per-image precision / recall at a given distance threshold
    (in heatmap-pixel units).
    """
    peaks = find_peaks(pred_heatmap, threshold=0.3, nms_radius=3)

    gt = np.array(gt_keypoints, dtype=np.float32)
    gt_hm = gt / stride

    if len(peaks) == 0:
        return 0.0, 0.0, 0
    if len(gt_hm) == 0:
        return 0.0, 1.0, len(peaks)

    pred_xy = np.array([[p[0], p[1]] for p in peaks], dtype=np.float32)

    matched_gt = set()
    matched_pred = set()
    for pi, pxy in enumerate(pred_xy):
        dists = np.linalg.norm(gt_hm - pxy, axis=1)
        best_gi = int(np.argmin(dists))
        if dists[best_gi] < dist_thresh and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    tp = len(matched_gt)
    precision = tp / len(peaks) if len(peaks) > 0 else 0.0
    recall = tp / len(gt_hm) if len(gt_hm) > 0 else 0.0
    return precision, recall, len(peaks)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    with open(args.annotations) as f:
        all_annots = json.load(f)

    available_images = set(os.listdir(args.images))
    all_files = sorted(f for f in all_annots if f in available_images)
    print(f"[TRAIN] Found {len(all_files)} annotated images")

    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_split))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"[TRAIN] Train: {len(train_files)}, Val: {len(val_files)}")

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
    print(f"[MODEL] {args.backbone} | params: {total_params:,} | trainable: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_steps), gamma=args.lr_factor
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets, _meta) in enumerate(train_loader):
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

        model.eval()
        val_loss = 0.0
        val_prec, val_rec = 0.0, 0.0

        with torch.no_grad():
            for images, targets, meta in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = model(images)
                val_loss += criterion(preds, targets).item()

                pred_np = torch.sigmoid(preds[0, 0]).cpu().numpy()
                gt_kps = meta["keypoints"]
                if isinstance(gt_kps, torch.Tensor):
                    gt_kps = gt_kps.numpy().tolist()
                elif isinstance(gt_kps[0], torch.Tensor):
                    gt_kps = [[k.item() for k in kp] for kp in gt_kps]
                elif isinstance(gt_kps[0], list):
                    gt_kps_flat = []
                    for i in range(len(gt_kps[0])):
                        kp_x = gt_kps[0][i].item() if isinstance(gt_kps[0][i], torch.Tensor) else gt_kps[0][i]
                        kp_y = gt_kps[1][i].item() if isinstance(gt_kps[1][i], torch.Tensor) else gt_kps[1][i]
                        gt_kps_flat.append([kp_x, kp_y])
                    gt_kps = gt_kps_flat

                p, r, _ = compute_detection_metrics(pred_np, gt_kps, stride=4)
                val_prec += p
                val_rec += r

        n_val_batches = max(len(val_loader), 1)
        avg_val_loss = val_loss / n_val_batches
        avg_prec = val_prec / n_val_batches
        avg_rec = val_rec / n_val_batches

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={avg_train_loss:.6f} | "
              f"val_loss={avg_val_loss:.6f} | "
              f"val_P={avg_prec:.3f} R={avg_rec:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.1e} | "
              f"{elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": {
                    "backbone": args.backbone,
                    "input_size": input_size,
                    "sigma": args.sigma,
                },
            }, ckpt_path)
            print(f"  → saved best model (val_loss={avg_val_loss:.6f})")

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

    print(f"\n[TRAIN] Done. Best val loss: {best_val_loss:.6f}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train vial keypoint heatmap model")
    p.add_argument("--annotations", type=str, required=True,
                   help="Path to vial_keypoints.json")
    p.add_argument("--images", type=str, required=True,
                   help="Directory containing images")
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-steps", type=int, nargs="+", default=[90, 140, 180])
    p.add_argument("--lr-factor", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--input-h", type=int, default=512)
    p.add_argument("--input-w", type=int, default=512)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
