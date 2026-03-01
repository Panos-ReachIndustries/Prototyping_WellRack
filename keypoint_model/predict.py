"""
Inference script for the vial keypoint model.

Usage:
    # Single image
    python -m keypoint_model.predict \
        --checkpoint checkpoints/best_model.pth \
        --image assets/testing_images/img_1.png \
        --output output_vis.png

    # Batch directory
    python -m keypoint_model.predict \
        --checkpoint checkpoints/best_model.pth \
        --image-dir assets/testing_images \
        --output-dir output_vis/

    # Programmatic usage (see VialKeypointPredictor class)
"""
import argparse
import os

import cv2
import numpy as np
import torch

from .config import ModelConfig
from .model import VialKeypointNet
from .utils import draw_keypoints, find_peaks, heatmap_to_image_coords


class VialKeypointPredictor:
    """
    Loads a trained checkpoint and runs inference on images.
    Returns vial centre coordinates in original image space.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, checkpoint_path: str, device: str | None = None,
                 threshold: float = 0.3, nms_radius: int = 5):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold
        self.nms_radius = nms_radius

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        self.input_size = cfg_dict.get("input_size", (512, 512))
        self.stride = 4

        model_cfg = ModelConfig(backbone=cfg_dict.get("backbone", "resnet18"))
        self.model = VialKeypointNet(model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> list[tuple[float, float, float]]:
        """
        Run keypoint detection on a BGR image.

        Returns:
            List of (x, y, confidence) in original image coordinates.
        """
        orig_h, orig_w = image.shape[:2]
        inp_h, inp_w = self.input_size

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (inp_w, inp_h))

        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        raw_out = self.model(tensor)
        heatmap = torch.sigmoid(raw_out[0, 0]).cpu().numpy()

        peaks = find_peaks(heatmap, threshold=self.threshold,
                           nms_radius=self.nms_radius)

        keypoints = heatmap_to_image_coords(
            peaks, self.stride, orig_h, orig_w, inp_h, inp_w
        )
        return keypoints

    def predict_and_visualize(self, image: np.ndarray) -> tuple:
        """
        Returns (keypoints, vis_image, heatmap_vis).
        """
        keypoints = self.predict(image)
        vis = draw_keypoints(image, keypoints, radius=4, color=(0, 255, 0))
        inp_h, inp_w = self.input_size
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (inp_w, inp_h))
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            raw_out = self.model(tensor)
        heatmap = torch.sigmoid(raw_out[0, 0]).cpu().numpy()
        hm_vis = (heatmap * 255).astype(np.uint8)
        hm_vis = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        hm_vis = cv2.resize(hm_vis, (image.shape[1], image.shape[0]))
        return keypoints, vis, hm_vis


def main():
    parser = argparse.ArgumentParser(description="Vial keypoint inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of images for batch inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single image")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch inference")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=5)
    args = parser.parse_args()

    predictor = VialKeypointPredictor(
        args.checkpoint, threshold=args.threshold, nms_radius=args.nms_radius
    )

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Cannot read {args.image}")
        kps, vis, hm_vis = predictor.predict_and_visualize(img)
        print(f"Detected {len(kps)} vial keypoints")
        for i, (x, y, c) in enumerate(kps):
            print(f"  [{i:2d}] x={x:.1f}, y={y:.1f}, conf={c:.3f}")

        out_path = args.output or "keypoint_output.png"
        combined = np.hstack([vis, hm_vis])
        cv2.imwrite(out_path, combined)
        print(f"Saved → {out_path}")

    elif args.image_dir:
        out_dir = args.output_dir or "keypoint_output"
        os.makedirs(out_dir, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg"}
        files = sorted(f for f in os.listdir(args.image_dir)
                       if os.path.splitext(f)[1].lower() in exts)
        for fname in files:
            img = cv2.imread(os.path.join(args.image_dir, fname))
            if img is None:
                continue
            kps, vis, hm_vis = predictor.predict_and_visualize(img)
            combined = np.hstack([vis, hm_vis])
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, combined)
            print(f"{fname}: {len(kps)} keypoints → {out_path}")
    else:
        parser.error("Provide --image or --image-dir")


if __name__ == "__main__":
    main()
