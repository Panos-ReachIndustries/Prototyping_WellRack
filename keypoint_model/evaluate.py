"""
Evaluate a trained vial keypoint model on the held-out test split.

Produces:
  - Per-image precision, recall, F1, detection count, mean distance error
  - Aggregate metrics across the test set
  - Visualisation images: detected keypoints + heatmap overlay + GT overlay

Usage:
    python -m keypoint_model.evaluate \
        --checkpoint checkpoints/best_model.pth \
        --annotations assets/testing_images/vial_keypoints.json \
        --images assets/testing_images \
        --splits assets/testing_images/splits.json \
        --output-dir eval_results
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from .config import ModelConfig
from .model import VialKeypointNet
from .utils import find_peaks, generate_heatmap


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr, input_size):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size[1], input_size[0]))
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)


def evaluate_image(model, image_bgr, gt_keypoints, input_size, device,
                   threshold=0.3, nms_radius=5, dist_thresh=4.0):
    """Run inference and compute metrics for a single image."""
    orig_h, orig_w = image_bgr.shape[:2]
    inp_h, inp_w = input_size
    stride = 4

    tensor = preprocess(image_bgr, input_size).to(device)
    with torch.no_grad():
        raw_out = model(tensor)
    heatmap = torch.sigmoid(raw_out[0, 0]).cpu().numpy()

    peaks = find_peaks(heatmap, threshold=threshold, nms_radius=nms_radius)

    sx = orig_w / inp_w
    sy = orig_h / inp_h
    pred_kps = []
    for (hx, hy, conf) in peaks:
        ix = (hx + 0.5) * stride * sx
        iy = (hy + 0.5) * stride * sy
        pred_kps.append((ix, iy, conf))

    gt = np.array(gt_keypoints, dtype=np.float32)

    matched_gt = set()
    matched_dists = []
    if len(pred_kps) > 0 and len(gt) > 0:
        for pi, (px, py, _) in enumerate(pred_kps):
            dists = np.sqrt((gt[:, 0] - px) ** 2 + (gt[:, 1] - py) ** 2)
            best_gi = int(np.argmin(dists))
            pix_thresh = dist_thresh * stride * max(sx, sy)
            if dists[best_gi] < pix_thresh and best_gi not in matched_gt:
                matched_gt.add(best_gi)
                matched_dists.append(float(dists[best_gi]))

    tp = len(matched_gt)
    fp = len(pred_kps) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_dist = float(np.mean(matched_dists)) if matched_dists else 0.0

    metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_pred": len(pred_kps), "n_gt": len(gt),
        "mean_dist_px": mean_dist,
    }
    return pred_kps, heatmap, metrics


def draw_evaluation(image_bgr, pred_kps, gt_keypoints, heatmap, input_size):
    """Create a side-by-side visualisation: detections | heatmap overlay."""
    orig_h, orig_w = image_bgr.shape[:2]
    vis = image_bgr.copy()

    for (gx, gy) in gt_keypoints:
        cv2.circle(vis, (int(gx), int(gy)), 5, (0, 255, 255), 1)

    for (px, py, conf) in pred_kps:
        cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 0), -1)

    cv2.putText(vis, f"Green=Pred ({len(pred_kps)})  Yellow=GT ({len(gt_keypoints)})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_resized = cv2.resize(hm_color, (orig_w, orig_h))
    overlay = cv2.addWeighted(image_bgr, 0.5, hm_resized, 0.5, 0)
    cv2.putText(overlay, "Heatmap overlay",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    combined = np.hstack([vis, overlay])
    return combined


def main():
    parser = argparse.ArgumentParser(description="Evaluate vial keypoint model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    input_size = tuple(cfg_dict.get("input_size", (384, 384)))
    model_cfg = ModelConfig(backbone=cfg_dict.get("backbone", "resnet18"),
                            pretrained=False)
    model = VialKeypointNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[EVAL] Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')}")
    print(f"[EVAL] Input size: {input_size}, backbone: {cfg_dict.get('backbone')}")

    with open(args.annotations) as f:
        annotations = json.load(f)
    with open(args.splits) as f:
        splits = json.load(f)

    test_files = splits["test"]
    print(f"[EVAL] Test images: {len(test_files)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []
    for fname in test_files:
        img_path = os.path.join(args.images, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  SKIP {fname}: cannot read")
            continue

        gt_kps = annotations.get(fname, [])
        pred_kps, heatmap, metrics = evaluate_image(
            model, img, gt_kps, input_size, device,
            threshold=args.threshold, nms_radius=args.nms_radius
        )
        all_metrics.append({"filename": fname, **metrics})

        vis = draw_evaluation(img, pred_kps, gt_kps, heatmap, input_size)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"{base}_eval.png")
        cv2.imwrite(out_path, vis)

        print(f"  {fname}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} | pred={metrics['n_pred']} gt={metrics['n_gt']} "
              f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
              f"dist={metrics['mean_dist_px']:.1f}px")

    # Aggregate
    if all_metrics:
        avg_p = np.mean([m["precision"] for m in all_metrics])
        avg_r = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])
        avg_dist = np.mean([m["mean_dist_px"] for m in all_metrics])
        total_tp = sum(m["tp"] for m in all_metrics)
        total_fp = sum(m["fp"] for m in all_metrics)
        total_fn = sum(m["fn"] for m in all_metrics)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        print(f"\n[EVAL] === Aggregate (macro-avg over {len(all_metrics)} images) ===")
        print(f"  Precision: {avg_p:.3f}")
        print(f"  Recall:    {avg_r:.3f}")
        print(f"  F1:        {avg_f1:.3f}")
        print(f"  Mean dist: {avg_dist:.1f}px")
        print(f"\n[EVAL] === Micro-avg ===")
        print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"  Precision: {micro_p:.3f}")
        print(f"  Recall:    {micro_r:.3f}")
        print(f"  F1:        {micro_f1:.3f}")

        summary = {
            "per_image": all_metrics,
            "macro_avg": {"precision": avg_p, "recall": avg_r, "f1": avg_f1,
                          "mean_dist_px": avg_dist},
            "micro_avg": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                          "precision": micro_p, "recall": micro_r, "f1": micro_f1},
        }
        summary_path = os.path.join(args.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[EVAL] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
