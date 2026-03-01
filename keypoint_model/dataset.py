"""
Dataset and augmentation pipeline for vial keypoint detection.

Annotation format (JSON):
{
  "img_1.png": [[x1, y1], [x2, y2], ...],
  "img_2.png": [[x1, y1], [x2, y2], ...],
  ...
}
Coordinates are in original image pixel space.
"""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import generate_heatmap


class VialKeypointDataset(Dataset):
    """
    Yields (image_tensor, heatmap_tensor, meta) for training / evaluation.

    image_tensor : (3, input_h, input_w)  float32, ImageNet-normalised
    heatmap_tensor : (1, hm_h, hm_w)     float32, Gaussian target
    meta : dict with original dims + keypoints for evaluation
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, annotations_path: str, images_dir: str,
                 input_size: tuple[int, int] = (384, 384),
                 heatmap_stride: int = 4,
                 sigma: float = 2.5,
                 augment: bool = False,
                 file_list: list[str] | None = None):
        with open(annotations_path) as f:
            self.all_annotations = json.load(f)

        self.images_dir = images_dir
        self.input_h, self.input_w = input_size
        self.hm_h = self.input_h // heatmap_stride
        self.hm_w = self.input_w // heatmap_stride
        self.stride = heatmap_stride
        self.sigma = sigma
        self.augment = augment

        if file_list is not None:
            self.file_names = [f for f in file_list if f in self.all_annotations]
        else:
            available = set(os.listdir(images_dir))
            self.file_names = sorted(
                f for f in self.all_annotations if f in available
            )

        if len(self.file_names) == 0:
            raise RuntimeError(
                f"No matching images found. "
                f"annotations keys: {list(self.all_annotations.keys())[:5]}, "
                f"images_dir: {images_dir}"
            )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        keypoints = np.array(self.all_annotations[fname], dtype=np.float32)

        if self.augment:
            img, keypoints = self._augment(img, keypoints)

        sx = self.input_w / img.shape[1]
        sy = self.input_h / img.shape[0]
        img_resized = cv2.resize(img, (self.input_w, self.input_h))

        kps_scaled = keypoints.copy()
        kps_scaled[:, 0] *= sx
        kps_scaled[:, 1] *= sy

        kps_heatmap = kps_scaled.copy()
        kps_heatmap[:, 0] /= self.stride
        kps_heatmap[:, 1] /= self.stride

        heatmap = generate_heatmap(self.hm_h, self.hm_w,
                                   kps_heatmap.tolist(), self.sigma)

        img_tensor = self._to_tensor(img_resized)
        hm_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        meta = {
            "filename": fname,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "keypoints": kps_scaled.tolist(),
        }
        return img_tensor, hm_tensor, meta

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1))

    # ------------------------------------------------------------------
    # Augmentation pipeline
    # ------------------------------------------------------------------
    def _augment(self, img: np.ndarray, kps: np.ndarray):
        h, w = img.shape[:2]

        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            kps[:, 0] = w - kps[:, 0]

        # Vertical flip
        if random.random() < 0.5:
            img = img[::-1, :].copy()
            kps[:, 1] = h - kps[:, 1]

        # Rotation ±15°
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            ones = np.ones((len(kps), 1), dtype=np.float32)
            kps_h = np.hstack([kps, ones])
            kps = (M @ kps_h.T).T

        # Scale jitter 0.8–1.2
        if random.random() < 0.5:
            scale = random.uniform(0.80, 1.20)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                img = cv2.resize(img, (new_w, new_h))
                kps[:, 0] *= scale
                kps[:, 1] *= scale

        # Translation shift ±5%
        if random.random() < 0.3:
            h2, w2 = img.shape[:2]
            tx = random.uniform(-0.05, 0.05) * w2
            ty = random.uniform(-0.05, 0.05) * h2
            M_t = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M_t, (w2, h2), borderMode=cv2.BORDER_REFLECT)
            kps[:, 0] += tx
            kps[:, 1] += ty

        # Brightness jitter
        if random.random() < 0.4:
            delta = random.randint(-40, 40)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        # Contrast jitter
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            mean_val = img.mean()
            img = np.clip((img.astype(np.float32) - mean_val) * factor + mean_val,
                          0, 255).astype(np.uint8)

        # Saturation / value jitter in HSV
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.6, 1.4)
            hsv[:, :, 2] *= random.uniform(0.7, 1.3)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Gaussian noise
        if random.random() < 0.25:
            noise_std = random.uniform(3, 15)
            noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Gaussian blur
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img, kps
