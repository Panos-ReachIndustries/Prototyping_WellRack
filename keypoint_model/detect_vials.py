"""
Drop-in replacement for the colour-based vial detector.

Provides a class with the same interface expected by the downstream
grid-generation stage: given a plate ROI image, return centroids.

Usage:
    from keypoint_model.detect_vials import KeypointVialsDetector

    detector = KeypointVialsDetector("checkpoints/best_model.pth")
    centroids, vis_frame = detector.detect(frame, plate_mask=mask)
"""
import cv2
import numpy as np
import torch

from .config import ModelConfig
from .model import VialKeypointNet
from .utils import draw_keypoints, find_peaks


class KeypointVialsDetector:
    """
    Neural-network based vial detector using heatmap regression.
    Replaces the HSL + KMeans + connected-components pipeline.
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
        self.input_size = tuple(cfg_dict.get("input_size", (512, 512)))
        self.stride = 4

        model_cfg = ModelConfig(
            backbone=cfg_dict.get("backbone", "resnet18"),
            pretrained=False,
        )
        self.model = VialKeypointNet(model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size[1], self.input_size[0]))
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)

    @torch.no_grad()
    def detect(self, frame: np.ndarray,
               plate_mask: np.ndarray | None = None
               ) -> tuple[list[list[float]], np.ndarray]:
        """
        Detect vial centres in a BGR plate ROI image.

        Args:
            frame: BGR image (plate crop or full frame).
            plate_mask: optional binary mask; peaks outside it are discarded.

        Returns:
            centroids: list of [x, y] positions in frame coordinates.
            vis_frame: frame with detected keypoints drawn.
        """
        orig_h, orig_w = frame.shape[:2]
        inp_h, inp_w = self.input_size

        tensor = self._preprocess(frame).to(self.device)
        raw_out = self.model(tensor)
        heatmap = torch.sigmoid(raw_out[0, 0]).cpu().numpy()

        if plate_mask is not None:
            mask_resized = cv2.resize(
                plate_mask.astype(np.uint8),
                (heatmap.shape[1], heatmap.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            mask_binary = (mask_resized > 0).astype(np.float32)
            heatmap = heatmap * mask_binary

        peaks = find_peaks(heatmap, threshold=self.threshold,
                           nms_radius=self.nms_radius)

        sx = orig_w / inp_w
        sy = orig_h / inp_h
        centroids = []
        kps_for_vis = []
        for (hx, hy, conf) in peaks:
            ix = (hx + 0.5) * self.stride * sx
            iy = (hy + 0.5) * self.stride * sy
            centroids.append([ix, iy])
            kps_for_vis.append((ix, iy, conf))

        vis_frame = draw_keypoints(frame, kps_for_vis, radius=4,
                                   color=(0, 255, 0))
        return centroids, vis_frame
