"""
visualize_with_gridtracker12_video.py
======================================
Video visualisation with GridTracker v12 using KeypointVialsDetector.

Uses the neural network-based keypoint model for initial vial detection,
replacing the HSL + KMeans + connected-components pipeline.

Improvements over v11:
- Uses KeypointVialsDetector (neural network heatmap regression) for vial detection
- More robust to lighting variations and different rack types
- Maintains all v11 improvements (FFT features, pairwise clustering, per-frame classification)
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Import WellRack modules
from gridtracker_functions_method_best import VialsDetector, compute_grid_confidence, get_plate_masks
from keypoint_model.detect_vials import KeypointVialsDetector

# Import LidStateClassifier
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "V2"))
from GridTracker.LidStateClassifier import LidStateClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def convert_windows_path_to_wsl(windows_path: str) -> str:
    normalized = windows_path.replace("\\\\", "\\").replace("/", "\\")
    
    if normalized.startswith("\\wsl.localhost\\Ubuntu\\") or normalized.startswith("\\wsl$\\Ubuntu\\"):
        wsl_path = normalized.replace("\\wsl.localhost\\Ubuntu\\", "/").replace("\\wsl$\\Ubuntu\\", "/")
        wsl_path = wsl_path.replace("\\", "/")
        return wsl_path
    elif "\\" in normalized:
        return normalized.replace("\\", "/")
    return windows_path


# Object colors for annotation overlay
OBJECT_COLORS = {
    "2": (255, 0, 0),
    "3": (0, 255, 0),
    "4": (0, 0, 255),
    "7": (0, 255, 255),
}

# Grid line colors
GRID_DOMINANT_COLOR = (0, 0, 255)
GRID_RECESSIVE_COLOR = (255, 0, 0)
GRID_POINT_COLOR = (0, 255, 0)


class LineTracker:
    """
    Spatial position-based line tracker that assigns persistent IDs based on
    position relative to a fixed plate origin (top-left corner).
    
    IDs are assigned by sorting lines by their position:
    - Dominant lines (vertical): sorted left-to-right (by x-position) -> D1, D2, ..., D8
    - Recessive lines (horizontal): sorted top-to-bottom (by y-position) -> R1, R2, ..., R12
    """

    def __init__(self, num_dominant: int = 8, num_recessive: int = 12):
        self.num_dominant = num_dominant
        self.num_recessive = num_recessive
        self.plate_origin = None
        self.plate_bbox = None

    def _compute_line_position(
        self, angle_deg: float, d: float, is_dominant: bool, plate_bbox: tuple[int, int, int, int]
    ) -> float:
        x, y, w, h = plate_bbox
        angle_rad = math.radians(angle_deg + 90)
        cos_t = math.cos(angle_rad)
        sin_t = math.sin(angle_rad)
        
        if is_dominant:
            if abs(cos_t) > 1e-6:
                x_at_top = (d - sin_t * y) / cos_t
                return x_at_top
            else:
                return x
        else:
            if abs(sin_t) > 1e-6:
                y_at_left = (d - cos_t * x) / sin_t
                return y_at_left
            else:
                return y

    def _update_plate_reference(self, plate_mask: np.ndarray | None):
        if plate_mask is None or not np.any(plate_mask > 0):
            return
        
        x, y, w, h = cv2.boundingRect(plate_mask)
        self.plate_bbox = (x, y, w, h)
        self.plate_origin = (x, y)

    def update(
        self,
        dominant_lines: list[tuple[tuple[float, float], Any]],
        recessive_lines: list[tuple[tuple[float, float], Any]],
        plate_mask: np.ndarray | None,
    ) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
        self._update_plate_reference(plate_mask)
        
        if self.plate_bbox is None:
            dominant_with_ids = [(line[0][0], line[0][1], f"D{i+1}") 
                                for i, line in enumerate(dominant_lines)]
            recessive_with_ids = [(line[0][0], line[0][1], f"R{i+1}") 
                                 for i, line in enumerate(recessive_lines)]
            return dominant_with_ids, recessive_with_ids
        
        dominant_with_ids = self._assign_spatial_ids(
            dominant_lines, is_dominant=True, id_prefix="D", max_ids=self.num_dominant
        )
        recessive_with_ids = self._assign_spatial_ids(
            recessive_lines, is_dominant=False, id_prefix="R", max_ids=self.num_recessive
        )
        return dominant_with_ids, recessive_with_ids

    def _assign_spatial_ids(
        self, lines: list[tuple[tuple[float, float], Any]], 
        is_dominant: bool, id_prefix: str, max_ids: int
    ) -> list[tuple[float, float, str]]:
        if not lines or self.plate_bbox is None:
            return []
        
        line_positions = []
        for line in lines:
            (angle, d), _ = line[:2]
            pos = self._compute_line_position(angle, d, is_dominant, self.plate_bbox)
            line_positions.append((pos, line))
        
        line_positions.sort(key=lambda x: x[0])
        
        result = []
        for i, (pos, line) in enumerate(line_positions[:max_ids]):
            (angle, d), _ = line[:2]
            result.append((angle, d, f"{id_prefix}{i+1}"))
        
        return result


class KeypointVialsDetectorWrapper(VialsDetector):
    """
    Wrapper that uses KeypointVialsDetector for vial detection but maintains
    the VialsDetector interface for grid inference.
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pth", 
                 keypoint_threshold: float = 0.3, keypoint_nms_radius: int = 5,
                 use_gpu: bool = True):
        # Initialize parent class for grid inference methods
        super().__init__(use_gpu=use_gpu)
        
        # Initialize keypoint detector
        self.keypoint_detector = KeypointVialsDetector(
            checkpoint_path=checkpoint_path,
            device="cuda" if use_gpu else "cpu",
            threshold=keypoint_threshold,
            nms_radius=keypoint_nms_radius
        )
        logger.info(f"KeypointVialsDetector initialized with checkpoint: {checkpoint_path}")

    def detect_and_track_vials(self, frame, reference_hist=None, frame_count=0):
        """
        Override to use KeypointVialsDetector for vial detection.
        Still uses parent class methods for grid inference.
        """
        try:
            import time
            orig_h, orig_w = frame.shape[:2]

            # Step 1: Crop to plate mask bounding box
            crop_x, crop_y = 0, 0
            if self.current_plate_mask is not None and np.any(self.current_plate_mask > 0):
                cx, cy, cw, ch = cv2.boundingRect(self.current_plate_mask)
                pad_x = max(10, int(cw * 0.05))
                pad_y = max(10, int(ch * 0.05))
                crop_x = max(0, cx - pad_x)
                crop_y = max(0, cy - pad_y)
                crop_x2 = min(orig_w, cx + cw + pad_x)
                crop_y2 = min(orig_h, cy + ch + pad_y)
                
                cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
                cropped_mask = self.current_plate_mask[crop_y:crop_y2, crop_x:crop_x2]
                crop_h, crop_w = cropped_frame.shape[:2]
                logger.info(
                    f"[CROP] Plate bbox ({cx},{cy},{cw},{ch}) -> crop ({crop_x},{crop_y})-({crop_x2},{crop_y2}) = {crop_w}x{crop_h}"
                )
            else:
                cropped_frame = frame
                cropped_mask = self.current_plate_mask
                crop_h, crop_w = orig_h, orig_w

            # Step 2: Normalize the cropped plate region to canonical width
            norm_frame, scale = self.normalize_rack(cropped_frame)
            inv_scale = 1.0 / scale
            norm_h, norm_w = norm_frame.shape[:2]
            logger.info(
                f"[NORMALIZE] {crop_w}x{crop_h} -> {norm_w}x{norm_h} (scale={scale:.3f})"
            )

            # Resize the cropped mask to match normalized resolution
            original_mask = self.current_plate_mask
            if cropped_mask is not None:
                norm_mask = cv2.resize(
                    cropped_mask, (norm_w, norm_h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                norm_mask = None

            try:
                # Temporarily set mask for keypoint detector
                self.current_plate_mask = norm_mask

                # Use KeypointVialsDetector for detection
                start_vials_detection = time.perf_counter()
                centroids, vis_frame = self.keypoint_detector.detect(norm_frame, plate_mask=norm_mask)
                vials_detection_time = time.perf_counter() - start_vials_detection
                self.last_vials_detection_time = vials_detection_time

                # Convert centroids format: from [[x, y], ...] to [(x, y), ...]
                if centroids:
                    centroids = [(int(x), int(y)) for x, y in centroids]
                else:
                    centroids = None

                # Use parent class method for grid inference
                start_grid_generation = time.perf_counter()
                final_points, filtered_lines, inferred_points, overlay_grid_lines_frame = self.infer_grid_points(
                    centroids, norm_frame, frame_count
                )
                grid_generation_time = time.perf_counter() - start_grid_generation
                self.last_grid_generation_time = grid_generation_time
            finally:
                self.current_plate_mask = original_mask

            total_time = vials_detection_time + grid_generation_time
            logger.info(
                f"Frame {frame_count}: Vials detection: {vials_detection_time*1000:.2f}ms, "
                f"Grid generation: {grid_generation_time*1000:.2f}ms, "
                f"Total: {total_time*1000:.2f}ms"
            )

            # Step 3: Map all results back to original full-frame coordinates
            if centroids is not None:
                centroids = [
                    (int(x * inv_scale) + crop_x, int(y * inv_scale) + crop_y) for x, y in centroids
                ]

            scaled_lines = []
            for line_data in filtered_lines:
                (angle, d), point_count, is_dominant = line_data
                angle_rad = math.radians(angle + 90)
                cos_t = math.cos(angle_rad)
                sin_t = math.sin(angle_rad)
                d_original = d * inv_scale + crop_x * cos_t + crop_y * sin_t
                scaled_lines.append(((angle, d_original), point_count, is_dominant))
            filtered_lines = scaled_lines

            if inferred_points:
                inferred_points = [
                    (int(x * inv_scale) + crop_x, int(y * inv_scale) + crop_y) for x, y in inferred_points
                ]

            vis_frame = cv2.resize(vis_frame, (orig_w, orig_h))
            overlay_grid_lines_frame = cv2.resize(overlay_grid_lines_frame, (orig_w, orig_h))

            # Create debug images dict (keypoint detector doesn't provide detailed debug images)
            debug_images = {
                'keypoint_vis': vis_frame.copy(),
                'overlay_grid': overlay_grid_lines_frame.copy(),
            }

            # Return format matching VialsDetector interface
            return (
                None,  # bboxes (not used)
                None,  # bboxes (duplicate, not used)
                vis_frame,
                debug_images,
                overlay_grid_lines_frame,
                filtered_lines,
                centroids,
                inferred_points,
            )
        except Exception as e:
            logger.error(f"Error occurred in detect_and_track_vials: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], frame.copy(), {}, frame.copy(), [], [], []


class GridTrackerWithExternalMask:
    """
    GridTracker that uses KeypointVialsDetector for vial detection.
    Maintains all v11 improvements for lid classification.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        keypoint_threshold: float = 0.3,
        keypoint_nms_radius: int = 5,
        enable_lid_classification: bool = True,
        lid_classifier_strip_width: int = 40,
        lid_classifier_buffer_size: int = 5,
        lid_classifier_vial_count_threshold: float = 0.20,
        lid_classifier_reference_state: str = "CLOSED_LID",
        lid_classifier_diff_threshold_method: str = "median_iqr",
        lid_classifier_iqr_multiplier: float = 0.8,
        lid_classifier_min_visual_threshold: float = 40.0,
        lid_classifier_use_fft_features: bool = True,
        lid_classifier_use_pairwise_clustering: bool = True,
        lid_classifier_min_feature_separation: float = 0.08,
        lid_classifier_profile_sigma_factor: float = 0.20,
    ):
        self.dominant_lines = []
        self.recessive_lines = []
        self.current_grid_points = []
        self.vials_grid_buffer = deque(maxlen=20)
        self.best_vials_entry = {
            "frameIdx": 0,
            "vialsDominantLines": [],
            "vialsRecessiveLines": [],
            "confidence": -1.0,
            "gridPoints": [],
            "inferredPoints": [],
            "frameData": None,
        }
        self.prev_plate_mask = None
        self.plate_orientation_angle = None
        self.detector = KeypointVialsDetectorWrapper(
            checkpoint_path=checkpoint_path,
            keypoint_threshold=keypoint_threshold,
            keypoint_nms_radius=keypoint_nms_radius
        )
        self.input_folder = None
        self.frame_count = 0
        self.line_tracker = LineTracker(num_dominant=8, num_recessive=12)

        # Lid state classification
        self.enable_lid_classification = enable_lid_classification
        self.lid_states = {}
        if self.enable_lid_classification:
            self.lid_classifier = LidStateClassifier(
                strip_width=lid_classifier_strip_width,
                buffer_size=lid_classifier_buffer_size,
                vial_count_threshold=lid_classifier_vial_count_threshold,
                reference_state=lid_classifier_reference_state,
                diff_threshold_method=lid_classifier_diff_threshold_method,
                iqr_multiplier=lid_classifier_iqr_multiplier,
                min_visual_threshold=lid_classifier_min_visual_threshold,
                use_fft_features=lid_classifier_use_fft_features,
                use_pairwise_clustering=lid_classifier_use_pairwise_clustering,
                min_feature_separation=lid_classifier_min_feature_separation,
                profile_sigma_factor=lid_classifier_profile_sigma_factor,
                debug=False,
            )
            logger.info(
                f"LidStateClassifier v12 initialized: "
                f"fft={lid_classifier_use_fft_features}, "
                f"gap_split={lid_classifier_use_pairwise_clustering}, "
                f"sigma={lid_classifier_profile_sigma_factor}, "
                f"buffer={lid_classifier_buffer_size}"
            )
        else:
            self.lid_classifier = None
        
        # Profiling statistics
        self.profiling_stats = {
            'vials_detection_times': [],
            'grid_generation_times': [],
            'total_times': [],
        }
        
        # Debug images storage
        self.last_debug_images = {}
        self.show_debug = False
        self.debug_window_name = None
        self.debug_output_folder = None
        self.debug_image_name = None
        
        # Grid update interval optimization
        self.frames_since_grid_update = 0
        self.grid_update_interval = 5

        # Cached drawing data
        self._cached_line_endpoints = []
        self._cached_plate_box = None
        
        logger.info("GridTrackerWithExternalMask v12 initialized with KeypointVialsDetector.")

    def _compute_average_line_length(self, lines: list, plate_mask: np.ndarray | None) -> float:
        """Compute average length of lines within the plate mask."""
        if not lines or plate_mask is None:
            return 0.0
        
        x, y, w, h = cv2.boundingRect(plate_mask)
        x2, y2 = x + w, y + h
        
        total_length = 0.0
        count = 0
        
        for line_data in lines:
            if isinstance(line_data, tuple) and len(line_data) >= 2:
                (angle_deg, d), _ = line_data[:2]
            else:
                continue
            
            angle_rad = math.radians(angle_deg + 90)
            cos_t = math.cos(angle_rad)
            sin_t = math.sin(angle_rad)
            
            intersections = []
            
            if abs(cos_t) > 1e-6:
                x_top = (d - sin_t * y) / cos_t
                if x <= x_top <= x2:
                    intersections.append((x_top, y))
                x_bottom = (d - sin_t * y2) / cos_t
                if x <= x_bottom <= x2:
                    intersections.append((x_bottom, y2))
            
            if abs(sin_t) > 1e-6:
                y_left = (d - cos_t * x) / sin_t
                if y <= y_left <= y2:
                    intersections.append((x, y_left))
                y_right = (d - cos_t * x2) / sin_t
                if y <= y_right <= y2:
                    intersections.append((x2, y_right))
            
            if len(intersections) >= 2:
                dx = intersections[1][0] - intersections[0][0]
                dy = intersections[1][1] - intersections[0][1]
                length = math.sqrt(dx * dx + dy * dy)
                total_length += length
                count += 1
        
        return total_length / count if count > 0 else 0.0

    def track_frame_with_mask(self, frame: np.ndarray, plate_mask: np.ndarray | None):
        """
        Process a frame with an externally provided plate mask.
        """
        self.frame_count += 1

        new_vials_entry = {
            "frameIdx": self.frame_count,
            "vialsDominantLines": [],
            "vialsRecessiveLines": [],
            "confidence": 0.0,
            "points": [],
            "inferredPoints": [],
            "frameData": None,
        }

        grid_was_updated = False

        if plate_mask is not None and np.any(plate_mask > 0):
            current_mask = plate_mask.astype(np.uint8)

            # Check overlap with previous mask
            overlap_ratio = 1.0
            if self.prev_plate_mask is not None:
                if current_mask.shape != self.prev_plate_mask.shape:
                    self.prev_plate_mask = cv2.resize(
                        self.prev_plate_mask,
                        (current_mask.shape[1], current_mask.shape[0]),
                    )
                intersection = np.logical_and(
                    current_mask > 0, self.prev_plate_mask > 0
                ).sum()
                union = np.logical_or(
                    current_mask > 0, self.prev_plate_mask > 0
                ).sum()
                overlap_ratio = intersection / union if union else 0
                if overlap_ratio < 0.1:
                    logger.debug(
                        f"Low overlap with previous mask ({overlap_ratio:.3f})"
                    )

            self.prev_plate_mask = current_mask.copy()
            self.detector.set_plate_mask(current_mask)
            self.plate_orientation_angle = getattr(self.detector, 'plate_orientation_angle', None)

            self.frames_since_grid_update += 1
            should_update_grid = (self.frames_since_grid_update >= self.grid_update_interval)
            
            if overlap_ratio < 0.5:
                should_update_grid = True
                self.frames_since_grid_update = 0
                logger.debug(f"Mask shifted significantly (overlap={overlap_ratio:.3f}), forcing grid update")
            
            if should_update_grid:
                self.frames_since_grid_update = 0
                grid_was_updated = True
                (
                    _,
                    _,
                    _,
                    debug_images,
                    _,
                    vials_filtered_lines,
                    vials_points,
                    vials_inferred_points,
                ) = self.detector.detect_and_track_vials(
                    frame, reference_hist=None, frame_count=self.frame_count
                )
            else:
                debug_images = None
                vials_filtered_lines = []
                vials_points = []
                vials_inferred_points = []
            
            if should_update_grid:
                if debug_images and isinstance(debug_images, dict):
                    self.last_debug_images = debug_images
                    logger.debug(f"Stored {len(debug_images)} debug images")
                    
                    if self.debug_output_folder is not None:
                        self.save_debug_images(self.debug_output_folder, self.frame_count, self.debug_image_name)
                
                if hasattr(self.detector, 'last_vials_detection_time'):
                    self.profiling_stats['vials_detection_times'].append(self.detector.last_vials_detection_time)
                if hasattr(self.detector, 'last_grid_generation_time'):
                    self.profiling_stats['grid_generation_times'].append(self.detector.last_grid_generation_time)
                if hasattr(self.detector, 'last_vials_detection_time') and hasattr(self.detector, 'last_grid_generation_time'):
                    total_time = self.detector.last_vials_detection_time + self.detector.last_grid_generation_time
                    self.profiling_stats['total_times'].append(total_time)

                # Separate lines into two groups
                group_dominant = []
                group_recessive = []
                logger.info(f"[DEBUG] Processing {len(vials_filtered_lines)} filtered lines")
                for line_data in vials_filtered_lines:
                    if not isinstance(line_data, tuple) or len(line_data) < 3:
                        logger.warning(f"[DEBUG] Skipping invalid line_data: {line_data}")
                        continue
                    (angle_deg, dist_val), points_dist, is_dominant = line_data
                    logger.info(f"[DEBUG] Line: angle={angle_deg:.1f}°, d={dist_val:.1f}, points={points_dist}, is_dominant={is_dominant}")
                    if is_dominant:
                        group_dominant.append(((angle_deg, dist_val), points_dist))
                    else:
                        group_recessive.append(((angle_deg, dist_val), points_dist))

                # Keep the group whose lines are parallel to the longest plate axis
                if group_dominant and group_recessive:
                    dom_avg_len = self._compute_average_line_length(group_dominant, current_mask)
                    rec_avg_len = self._compute_average_line_length(group_recessive, current_mask)
                    if rec_avg_len > dom_avg_len:
                        logger.info(f"Swapping: recessive lines are longer ({rec_avg_len:.1f}px > {dom_avg_len:.1f}px)")
                        vials_dominant_lines = group_recessive
                    else:
                        vials_dominant_lines = group_dominant
                else:
                    vials_dominant_lines = group_dominant if group_dominant else group_recessive

                logger.info(f"[DEBUG] Extracted {len(vials_dominant_lines)} dominant lines for storage")

                # Compute confidence
                vials_grid_conf = compute_grid_confidence(
                    vials_points,
                    vials_inferred_points,
                    vials_dominant_lines,
                    frame.shape,
                )

                new_vials_entry.update(
                    {
                        "vialsDominantLines": vials_dominant_lines,
                        "vialsRecessiveLines": [],
                        "confidence": vials_grid_conf,
                        "points": vials_points,
                        "inferredPoints": vials_inferred_points,
                    }
                )

                self.vials_grid_buffer.append(new_vials_entry)
                logger.debug(
                    f"Frame {self.frame_count}: confidence={vials_grid_conf:.2f} (grid updated)"
                )
            else:
                logger.debug(
                    f"Frame {self.frame_count}: reusing previous grid (skip update)"
                )
        else:
            logger.debug(f"No valid plate mask in frame #{self.frame_count}")

        # Update line IDs and cached drawing data when grid was recomputed
        if grid_was_updated:
            if len(self.vials_grid_buffer) > 0:
                best_entry = max(self.vials_grid_buffer, key=lambda e: e.get("confidence", -1.0))
                if best_entry.get("confidence", -1.0) > self.best_vials_entry.get("confidence", -1.0):
                    self.best_vials_entry = best_entry.copy()

            if self.best_vials_entry.get("vialsDominantLines"):
                dominant_with_ids, recessive_with_ids = self.line_tracker.update(
                    self.best_vials_entry["vialsDominantLines"],
                    self.best_vials_entry.get("vialsRecessiveLines", []),
                    self.prev_plate_mask,
                )
                self.dominant_lines = dominant_with_ids
                self.recessive_lines = recessive_with_ids
                self.current_grid_points = self.best_vials_entry.get("gridPoints", [])

        # Per-frame lid classification (Fix D)
        if self.enable_lid_classification and self.lid_classifier is not None:
            if self.dominant_lines and self.prev_plate_mask is not None:
                try:
                    self.lid_states = self.lid_classifier.classify_frame(
                        frame, self.dominant_lines, self.prev_plate_mask
                    )
                except Exception as e:
                    logger.warning(f"Lid classification failed: {e}")

    def get_confidence(self) -> float:
        """Get current grid confidence."""
        return self.best_vials_entry.get("confidence", 0.0)

    def draw_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw grid overlay on frame."""
        overlay = frame.copy()
        
        if self.prev_plate_mask is not None:
            x, y, w, h = cv2.boundingRect(self.prev_plate_mask)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Draw dominant lines
        for angle, d, line_id in self.dominant_lines:
            angle_rad = math.radians(angle + 90)
            cos_t = math.cos(angle_rad)
            sin_t = math.sin(angle_rad)
            
            h, w = frame.shape[:2]
            if abs(cos_t) > 1e-6:
                x1 = (d - sin_t * 0) / cos_t
                x2 = (d - sin_t * h) / cos_t
                if 0 <= x1 <= w:
                    cv2.line(overlay, (int(x1), 0), (int(x1), h), GRID_DOMINANT_COLOR, 2)
                if 0 <= x2 <= w:
                    cv2.line(overlay, (int(x2), 0), (int(x2), h), GRID_DOMINANT_COLOR, 2)
            
            # Draw line ID
            if self.prev_plate_mask is not None:
                px, py, pw, ph = cv2.boundingRect(self.prev_plate_mask)
                text_x = int((d - sin_t * py) / cos_t) if abs(cos_t) > 1e-6 else px
                text_y = py + 20
                cv2.putText(overlay, line_id, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_DOMINANT_COLOR, 2)

        # Draw grid points
        for point in self.current_grid_points:
            if len(point) >= 2:
                px, py = int(point[0]), int(point[1])
                cv2.circle(overlay, (px, py), 3, GRID_POINT_COLOR, -1)

        # Draw lid states
        if self.lid_states:
            for line_id, state in self.lid_states.items():
                color = (0, 255, 0) if state == "OPEN_LID" else (0, 0, 255)
                # Find line and draw state indicator
                for angle, d, lid in self.dominant_lines:
                    if lid == line_id:
                        angle_rad = math.radians(angle + 90)
                        cos_t = math.cos(angle_rad)
                        sin_t = math.sin(angle_rad)
                        h, w = frame.shape[:2]
                        if self.prev_plate_mask is not None:
                            px, py, pw, ph = cv2.boundingRect(self.prev_plate_mask)
                            text_x = int((d - sin_t * py) / cos_t) if abs(cos_t) > 1e-6 else px
                            text_y = py + 40
                            cv2.putText(overlay, state[:4], (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        break

        return overlay

    def save_debug_images(self, output_folder: str, frame_count: int, image_name: str | None = None):
        """Save debug images to folder."""
        if not self.last_debug_images:
            return
        
        os.makedirs(output_folder, exist_ok=True)
        base_name = image_name if image_name else f"frame_{frame_count:06d}"
        
        for key, img in self.last_debug_images.items():
            if isinstance(img, np.ndarray):
                filename = f"{base_name}_{key}.png"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, img)

    def toggle_debug_display(self):
        """Toggle debug display mode."""
        self.show_debug = not self.show_debug

    def show_debug_images(self, window_name: str = "Debug"):
        """Show debug images in OpenCV windows."""
        if not self.last_debug_images:
            return
        
        for key, img in self.last_debug_images.items():
            if isinstance(img, np.ndarray):
                cv2.imshow(f"{window_name}_{key}", img)


def process_video(
    video_path: str,
    output_path: str | None = None,
    checkpoint_path: str = "checkpoints/best_model.pth",
    keypoint_threshold: float = 0.3,
    keypoint_nms_radius: int = 5,
    show: bool = True,
    enable_lid_classification: bool = True,
    lid_classifier_strip_width: int = 40,
    lid_classifier_buffer_size: int = 5,
    lid_classifier_vial_count_threshold: float = 0.20,
    lid_classifier_reference_state: str = "CLOSED_LID",
    lid_classifier_use_fft_features: bool = True,
    lid_classifier_use_pairwise_clustering: bool = True,
    lid_classifier_min_feature_separation: float = 0.08,
    lid_classifier_profile_sigma_factor: float = 0.20,
) -> None:
    """Process video with GridTracker v12."""
    video_path = convert_windows_path_to_wsl(video_path)
    if output_path:
        output_path = convert_windows_path_to_wsl(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Output video: {output_path}")
    else:
        out = None

    grid_tracker = GridTrackerWithExternalMask(
        checkpoint_path=checkpoint_path,
        keypoint_threshold=keypoint_threshold,
        keypoint_nms_radius=keypoint_nms_radius,
        enable_lid_classification=enable_lid_classification,
        lid_classifier_strip_width=lid_classifier_strip_width,
        lid_classifier_buffer_size=lid_classifier_buffer_size,
        lid_classifier_vial_count_threshold=lid_classifier_vial_count_threshold,
        lid_classifier_reference_state=lid_classifier_reference_state,
        lid_classifier_use_fft_features=lid_classifier_use_fft_features,
        lid_classifier_use_pairwise_clustering=lid_classifier_use_pairwise_clustering,
        lid_classifier_min_feature_separation=lid_classifier_min_feature_separation,
        lid_classifier_profile_sigma_factor=lid_classifier_profile_sigma_factor,
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        logger.info(f"Processing frame {frame_count}/{total_frames}")

        # Get plate mask (using get_plate_masks from gridtracker_functions)
        plate_mask = get_plate_masks(frame)
        if plate_mask is not None and np.any(plate_mask > 0):
            grid_tracker.track_frame_with_mask(frame, plate_mask)
        else:
            grid_tracker.track_frame_with_mask(frame, None)

        annotated_frame = grid_tracker.draw_grid_overlay(frame)

        if out:
            out.write(annotated_frame)

        if show:
            cv2.imshow("WellRack GridTracker v12", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    if show:
        cv2.destroyAllWindows()

    logger.info(f"Processed {frame_count} frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize video with WellRack GridTracker v12 (KeypointVialsDetector)."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to keypoint model checkpoint (default: checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="Keypoint detection threshold (default: 0.3).",
    )
    parser.add_argument(
        "--keypoint-nms-radius",
        type=int,
        default=5,
        help="NMS radius for keypoint detection (default: 5).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display video during processing.",
    )
    parser.add_argument(
        "--no-lid-classification",
        action="store_true",
        help="Disable lid state classification.",
    )
    parser.add_argument(
        "--lid-classifier-strip-width",
        type=int,
        default=40,
        help="Lid classifier strip width (default: 40).",
    )
    parser.add_argument(
        "--lid-classifier-buffer-size",
        type=int,
        default=5,
        help="Lid classifier buffer size (default: 5).",
    )
    parser.add_argument(
        "--lid-classifier-vial-count-threshold",
        type=float,
        default=0.20,
        help="Lid classifier vial count threshold (default: 0.20).",
    )
    parser.add_argument(
        "--lid-classifier-reference-state",
        type=str,
        default="CLOSED_LID",
        choices=["OPEN_LID", "CLOSED_LID"],
        help="Reference state for lid classification (default: CLOSED_LID).",
    )
    parser.add_argument(
        "--no-fft-features",
        action="store_true",
        help="Disable FFT texture features.",
    )
    parser.add_argument(
        "--no-pairwise-clustering",
        action="store_true",
        help="Disable pairwise K-Means clustering.",
    )
    parser.add_argument(
        "--lid-classifier-profile-sigma-factor",
        type=float,
        default=0.20,
        help="Gaussian profile sigma factor (default: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        keypoint_threshold=args.keypoint_threshold,
        keypoint_nms_radius=args.keypoint_nms_radius,
        show=not args.no_show,
        enable_lid_classification=not args.no_lid_classification,
        lid_classifier_strip_width=args.lid_classifier_strip_width,
        lid_classifier_buffer_size=args.lid_classifier_buffer_size,
        lid_classifier_vial_count_threshold=args.lid_classifier_vial_count_threshold,
        lid_classifier_reference_state=args.lid_classifier_reference_state,
        lid_classifier_use_fft_features=not args.no_fft_features,
        lid_classifier_use_pairwise_clustering=not args.no_pairwise_clustering,
        lid_classifier_min_feature_separation=0.08,
        lid_classifier_profile_sigma_factor=args.lid_classifier_profile_sigma_factor,
    )


if __name__ == "__main__":
    main()
