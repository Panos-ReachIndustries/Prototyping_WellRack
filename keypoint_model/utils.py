"""
Heatmap generation and peak detection utilities.
"""
import numpy as np
import cv2


def generate_heatmap(height: int, width: int, keypoints: list[list[int]],
                     sigma: float = 2.5) -> np.ndarray:
    """
    Render a (H, W) heatmap with Gaussian blobs at each keypoint.

    Args:
        height, width: spatial dimensions of the heatmap.
        keypoints: list of [x, y] positions in *heatmap* coordinates.
        sigma: Gaussian std-dev in heatmap pixels.

    Returns:
        heatmap: float32 array in [0, 1] with shape (H, W).
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = int(3 * sigma + 0.5)

    for kp in keypoints:
        cx, cy = int(round(kp[0])), int(round(kp[1]))

        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        gx = np.arange(x0, x1) - cx
        gy = np.arange(y0, y1) - cy
        gx, gy = np.meshgrid(gx, gy)
        g = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))

        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)

    return heatmap


def find_peaks(heatmap: np.ndarray, threshold: float = 0.3,
               nms_radius: int = 5) -> list[tuple[float, float, float]]:
    """
    Extract keypoint locations from a heatmap via thresholding + NMS.

    Args:
        heatmap: 2-D float array (H, W) with values in [0, 1].
        threshold: minimum activation to consider a peak.
        nms_radius: suppress peaks within this pixel radius.

    Returns:
        List of (x, y, confidence) tuples in heatmap coordinates,
        sorted by confidence descending.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {heatmap.shape}")

    kernel = 2 * nms_radius + 1
    dilated = cv2.dilate(heatmap, np.ones((kernel, kernel), np.float32))
    peaks_mask = (heatmap == dilated) & (heatmap >= threshold)

    ys, xs = np.where(peaks_mask)
    confs = heatmap[ys, xs]

    order = np.argsort(-confs)
    results = []
    suppressed = np.zeros(len(order), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        x, y, c = float(xs[idx]), float(ys[idx]), float(confs[idx])
        results.append((x, y, c))

        dists_sq = (xs - x) ** 2 + (ys - y) ** 2
        suppressed |= dists_sq < nms_radius ** 2

    return results


def heatmap_to_image_coords(peaks: list[tuple[float, float, float]],
                            stride: int, orig_h: int, orig_w: int,
                            input_h: int, input_w: int
                            ) -> list[tuple[float, float, float]]:
    """
    Map peak coordinates from heatmap space back to original image space.

    peak_in_heatmap → ×stride → in model-input space → scale to original image.
    """
    sx = orig_w / input_w
    sy = orig_h / input_h

    result = []
    for (hx, hy, conf) in peaks:
        ix = (hx + 0.5) * stride * sx
        iy = (hy + 0.5) * stride * sy
        result.append((ix, iy, conf))
    return result


def draw_keypoints(image: np.ndarray,
                   keypoints: list[tuple[float, float, float]],
                   radius: int = 4,
                   color: tuple = (0, 255, 0),
                   thickness: int = -1) -> np.ndarray:
    """Draw detected keypoints on an image."""
    vis = image.copy()
    for (x, y, conf) in keypoints:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), radius, color, thickness)
        cv2.putText(vis, f"{conf:.2f}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return vis
