"""

--- loading the yolo_best_v11n.pt model




Script to visualize video with annotations and WellRack GridTracker overlay.

This script loads a video and annotation file, runs the WellRack GridTracker
using the plate annotations (object ID 4) as the detection mask, and overlays
the detected grid along with all other annotations.
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

# Import WellRack modules - adjust path if needed
from gridtracker_functions_method_best import VialsDetector, compute_grid_confidence, get_plate_masks
# from ultralytics import YOLO
# import torch

# Import LidStateClassifier
import sys
from pathlib import Path
# Add parent directory to path to import from V2
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
    # Normalize the path - handle both single and double backslashes
    normalized = windows_path.replace("\\\\", "\\").replace("/", "\\")
    
    # Handle Windows UNC path format for WSL
    if normalized.startswith("\\wsl.localhost\\Ubuntu\\") or normalized.startswith("\\wsl$\\Ubuntu\\"):
        # Convert to WSL path: \wsl.localhost\Ubuntu\home\user -> /home/user
        wsl_path = normalized.replace("\\wsl.localhost\\Ubuntu\\", "/").replace("\\wsl$\\Ubuntu\\", "/")
        wsl_path = wsl_path.replace("\\", "/")
        return wsl_path
    elif "\\" in normalized:
        # Convert backslashes to forward slashes for regular Windows paths
        return normalized.replace("\\", "/")
    return windows_path


# Object colors for annotation overlay
OBJECT_COLORS = {
    "2": (255, 0, 0),    # Blue
    "3": (0, 255, 0),    # Green
    "4": (0, 0, 255),    # Red - Plate
    "7": (0, 255, 255),  # Yellow
}

# Grid line colors
GRID_DOMINANT_COLOR = (0, 0, 255)    # Red for dominant (longest) lines
GRID_RECESSIVE_COLOR = (255, 0, 0)   # Blue for recessive (shorter/perpendicular) lines
GRID_POINT_COLOR = (0, 255, 0)       # Green for grid points


class LineTracker:
    """
    Spatial position-based line tracker that assigns persistent IDs based on
    position relative to a fixed plate origin (top-left corner).
    
    IDs are assigned by sorting lines by their position:
    - Dominant lines (vertical): sorted left-to-right (by x-position) -> D1, D2, ..., D8
    - Recessive lines (horizontal): sorted top-to-bottom (by y-position) -> R1, R2, ..., R12
    """

    def __init__(self, num_dominant: int = 8, num_recessive: int = 12):
        """
        Args:
            num_dominant: Expected number of dominant lines (default: 8 for 8x12 rack)
            num_recessive: Expected number of recessive lines (default: 12 for 8x12 rack)
        """
        self.num_dominant = num_dominant
        self.num_recessive = num_recessive
        self.plate_origin = None  # (x, y) - top-left corner of plate
        self.plate_bbox = None  # (x, y, w, h) - bounding box of plate

    def _compute_line_position(
        self, angle_deg: float, d: float, is_dominant: bool, plate_bbox: tuple[int, int, int, int]
    ) -> float:
        """
        Compute the position of a line relative to the plate origin.
        
        For dominant lines (vertical): return x-coordinate at top of plate (y = plate_top)
        For recessive lines (horizontal): return y-coordinate at left of plate (x = plate_left)
        
        Args:
            angle_deg: Line angle in degrees
            d: Line distance parameter (normal form: x*cos + y*sin = d)
            is_dominant: True for dominant (vertical) lines, False for recessive (horizontal)
            plate_bbox: (x, y, w, h) bounding box of plate
        
        Returns:
            Position value for sorting (x for dominant, y for recessive)
        """
        x, y, w, h = plate_bbox
        angle_rad = math.radians(angle_deg + 90)
        cos_t = math.cos(angle_rad)
        sin_t = math.sin(angle_rad)
        
        if is_dominant:
            # For vertical lines, get x-coordinate where line intersects top edge of plate
            # Line equation: x*cos_t + y*sin_t = d
            # At y = plate_top: x = (d - y*sin_t) / cos_t
            if abs(cos_t) > 1e-6:
                x_at_top = (d - sin_t * y) / cos_t
                return x_at_top
            else:
                # Line is nearly vertical, use x from distance parameter
                return x
        else:
            # For horizontal lines, get y-coordinate where line intersects left edge of plate
            # Line equation: x*cos_t + y*sin_t = d
            # At x = plate_left: y = (d - x*cos_t) / sin_t
            if abs(sin_t) > 1e-6:
                y_at_left = (d - cos_t * x) / sin_t
                return y_at_left
            else:
                # Line is nearly horizontal, use y from distance parameter
                return y

    def _update_plate_reference(self, plate_mask: np.ndarray | None):
        """Update the plate origin and bounding box reference from current frame."""
        if plate_mask is None or not np.any(plate_mask > 0):
            return
        
        x, y, w, h = cv2.boundingRect(plate_mask)
        self.plate_bbox = (x, y, w, h)
        self.plate_origin = (x, y)  # Top-left corner

    def update(
        self,
        dominant_lines: list[tuple[tuple[float, float], Any]],
        recessive_lines: list[tuple[tuple[float, float], Any]],
        plate_mask: np.ndarray | None,
    ) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
        """
        Update tracking with new detections and return lines with IDs based on spatial position.
        
        Args:
            dominant_lines: List of ((angle, dist), metadata) for dominant lines
            recessive_lines: List of ((angle, dist), metadata) for recessive lines
            plate_mask: Binary mask of the plate region
        
        Returns:
            (dominant_with_ids, recessive_with_ids) where each is list of (angle, dist, id_str)
        """
        # Update plate reference
        self._update_plate_reference(plate_mask)
        
        if self.plate_bbox is None:
            # No plate reference yet, return without IDs
            dominant_with_ids = [(line[0][0], line[0][1], f"D{i+1}") 
                                for i, line in enumerate(dominant_lines)]
            recessive_with_ids = [(line[0][0], line[0][1], f"R{i+1}") 
                                 for i, line in enumerate(recessive_lines)]
            return dominant_with_ids, recessive_with_ids
        
        # Assign IDs based on spatial position
        dominant_with_ids = self._assign_spatial_ids(
            dominant_lines, is_dominant=True, id_prefix="D", max_ids=self.num_dominant
        )
        recessive_with_ids = self._assign_spatial_ids(
            recessive_lines, is_dominant=False, id_prefix="R", max_ids=self.num_recessive
        )
        
        return dominant_with_ids, recessive_with_ids

    def _assign_spatial_ids(
        self,
        detected_lines: list[tuple[tuple[float, float], Any]],
        is_dominant: bool,
        id_prefix: str,
        max_ids: int,
    ) -> list[tuple[float, float, str]]:
        """
        Assign IDs to lines based on their spatial position relative to plate origin.
        
        Args:
            detected_lines: List of ((angle, dist), metadata)
            is_dominant: True for dominant (vertical) lines
            id_prefix: "D" or "R"
            max_ids: Maximum number of expected lines
        
        Returns:
            List of (angle, dist, id_str) sorted by position
        """
        if not detected_lines or self.plate_bbox is None:
            return []
        
        # Compute position for each line and sort
        lines_with_positions = []
        for line_data in detected_lines:
            (angle_deg, d), _ = line_data[:2]
            position = self._compute_line_position(angle_deg, d, is_dominant, self.plate_bbox)
            lines_with_positions.append((angle_deg, d, position))
        
        # Sort by position (left-to-right for dominant, top-to-bottom for recessive)
        lines_with_positions.sort(key=lambda x: x[2])
        
        # Assign IDs 1, 2, 3, ... based on sorted position
        results = []
        for idx, (angle_deg, d, _) in enumerate(lines_with_positions):
            # Use 1-based indexing, but cap at max_ids
            line_id = min(idx + 1, max_ids)
            id_str = f"{id_prefix}{line_id}"
            results.append((angle_deg, d, id_str))
        
        return results


def load_annotations(json_path: str) -> dict[str, Any]:
    """Load annotations from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def polygon_to_bbox(poly: list[list[int]]) -> tuple[int, int, int, int] | None:
    """Convert polygon points to bounding box (x, y, w, h)."""
    if not poly:
        return None
    pts = np.array(poly, dtype=np.int32)
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())
    x_max = int(pts[:, 0].max())
    y_max = int(pts[:, 1].max())
    return x_min, y_min, x_max - x_min, y_max - y_min


def polygon_to_mask(
    poly: list[list[int]], frame_shape: tuple[int, int]
) -> np.ndarray:
    """Convert polygon to binary mask."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if poly:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_bbox_and_id(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    obj_id: str,
    color: tuple[int, int, int],
) -> None:
    """Draw bounding box and object ID label."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"ID {obj_id}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_text = max(y, th + baseline + 2)
    cv2.rectangle(
        frame,
        (x, y_text - th - baseline - 4),
        (x + tw + 4, y_text),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (x + 2, y_text - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def draw_frame_polygons(
    frame: np.ndarray,
    frame_ann: dict[str, Any],
    alpha: float = 0.3,
    skip_ids: set[str] | None = None,
) -> None:
    """Draw annotation polygons on the frame with transparency."""
    if skip_ids is None:
        skip_ids = set()

    overlay = frame.copy()

    for obj_id, obj_data in frame_ann.items():
        if obj_id in skip_ids:
            continue

        polygons = obj_data.get("polygons", [])
        if not polygons:
            continue

        color = OBJECT_COLORS.get(obj_id, (255, 255, 255))

        for poly in polygons:
            if not poly:
                continue

            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [pts], color=color)

            bbox = polygon_to_bbox(poly)
            if bbox is not None:
                draw_bbox_and_id(overlay, bbox, obj_id, color)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def get_plate_mask_from_annotations(
    frame_ann: dict[str, Any],
    frame_shape: tuple[int, int],
    plate_id: str = "4",
) -> np.ndarray | None:
    """
    Extract plate mask from annotations.

    Args:
        frame_ann: Annotations for this frame
        frame_shape: (height, width) of the frame
        plate_id: Object ID for the plate in annotations

    Returns:
        Binary mask or None if no plate annotation found
    """
    if plate_id not in frame_ann:
        return None

    obj_data = frame_ann[plate_id]
    polygons = obj_data.get("polygons", [])

    if not polygons:
        return None

    # Combine all plate polygons into one mask
    combined_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for poly in polygons:
        if poly:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(combined_mask, [pts], 255)

    return combined_mask if np.any(combined_mask > 0) else None


class GridTrackerWithExternalMask:
    """
    A modified GridTracker that accepts external plate masks from user-provided bounding boxes.
    """

    def __init__(
        self,
        enable_lid_classification: bool = True,
        lid_classifier_strip_width: int = 40,
        lid_classifier_buffer_size: int = 5,
        lid_classifier_vial_count_threshold: float = 0.5,
        lid_classifier_reference_state: str = "CLOSED_LID",
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
        self.plate_orientation_angle = None  # Store plate orientation
        self.detector = VialsDetector()
        self.input_folder = None  # Will be set when processing images
        self.frame_count = 0
        self.line_tracker = LineTracker(num_dominant=8, num_recessive=12)
        
        # Lid state classification
        self.enable_lid_classification = enable_lid_classification
        self.lid_states = {}  # {column_id: "OPEN_LID" or "CLOSED_LID"}
        if self.enable_lid_classification:
            self.lid_classifier = LidStateClassifier(
                strip_width=lid_classifier_strip_width,
                buffer_size=lid_classifier_buffer_size,
                vial_count_threshold=lid_classifier_vial_count_threshold,
                reference_state=lid_classifier_reference_state,
                debug=False,
            )
            logger.info("LidStateClassifier initialized")
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
        self.show_debug = False  # Toggle for debug display
        self.debug_window_name = None  # Store current debug window name
        self.debug_output_folder = None  # Folder to save debug images
        self.debug_image_name = None  # Original image name for debug filename
        
        # Grid update interval optimization
        self.frames_since_grid_update = 0
        self.grid_update_interval = 5  # Only recompute grid every 3 frames

        # Cached drawing data (recomputed only when grid updates)
        self._cached_line_endpoints = []  # list of (p1, p2, line_id, text_x, text_y, text_w, text_h)
        self._cached_plate_box = None  # oriented bounding box points
        
        logger.info("GridTrackerWithExternalMask initialized with spatial position-based LineTracker.")


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
            
            # Find intersections with bounding box
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
                # Compute length between first two intersections
                dx = intersections[1][0] - intersections[0][0]
                dy = intersections[1][1] - intersections[0][1]
                length = math.sqrt(dx * dx + dy * dy)
                total_length += length
                count += 1
        
        return total_length / count if count > 0 else 0.0

    def track_frame_with_mask(self, frame: np.ndarray, plate_mask: np.ndarray | None):
        """
        Process a frame with an externally provided plate mask.

        Args:
            frame: BGR uint8 numpy array
            plate_mask: Binary mask (uint8) where plate region is non-zero,
                       or None if no plate in this frame
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

            # Check overlap with previous mask for temporal consistency
            overlap_ratio = 1.0  # Default to full overlap if no previous mask
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

            # CRITICAL: Always update mask every frame (rack may shift)
            # This must happen BEFORE any grid computation decisions
            self.prev_plate_mask = current_mask.copy()
            self.detector.set_plate_mask(current_mask)
            # Store the plate orientation from the detector (if available)
            self.plate_orientation_angle = getattr(self.detector, 'plate_orientation_angle', None)

            # Only update grid computation every N frames to reduce computation
            # Note: Mask is always updated above, but expensive grid detection is skipped
            self.frames_since_grid_update += 1
            should_update_grid = (self.frames_since_grid_update >= self.grid_update_interval)
            
            # If mask changed significantly, force grid update
            if overlap_ratio < 0.5:
                should_update_grid = True
                self.frames_since_grid_update = 0
                logger.debug(f"Mask shifted significantly (overlap={overlap_ratio:.3f}), forcing grid update")
            
            if should_update_grid:
                self.frames_since_grid_update = 0
                grid_was_updated = True
                # Detect vials grid using the provided mask (profiling is done inside)
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
                # Reuse previous grid lines - skip detection
                debug_images = None
                vials_filtered_lines = []
                vials_points = []
                vials_inferred_points = []
            
            # Store debug images for real-time viewing (only if grid was updated)
            if should_update_grid:
                if debug_images and isinstance(debug_images, dict):
                    self.last_debug_images = debug_images
                    logger.debug(f"Stored {len(debug_images)} debug images (keys: {list(debug_images.keys())[:5]}...)")
                    
                    # Automatically save debug images if output folder is set
                    if self.debug_output_folder is not None:
                        self.save_debug_images(self.debug_output_folder, self.frame_count, self.debug_image_name)
                    else:
                        logger.debug(f"Debug images available but not saved (debug_output_folder is None). Use --debug-output-folder to save them.")
                else:
                    if debug_images is None:
                        logger.warning("No debug images returned from detection method")
                    else:
                        logger.debug(f"Debug images not available or invalid type: {type(debug_images)}")
                
                # Collect profiling statistics
                if hasattr(self.detector, 'last_vials_detection_time'):
                    self.profiling_stats['vials_detection_times'].append(self.detector.last_vials_detection_time)
                if hasattr(self.detector, 'last_grid_generation_time'):
                    self.profiling_stats['grid_generation_times'].append(self.detector.last_grid_generation_time)
                if hasattr(self.detector, 'last_vials_detection_time') and hasattr(self.detector, 'last_grid_generation_time'):
                    total_time = self.detector.last_vials_detection_time + self.detector.last_grid_generation_time
                    self.profiling_stats['total_times'].append(total_time)

                # Separate lines into two groups by detector classification
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
                # (i.e. the group with greater average line length within the mask)
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

                # Compute confidence (only using dominant lines)
                vials_grid_conf = compute_grid_confidence(
                    vials_points,
                    vials_inferred_points,
                    vials_dominant_lines,
                    frame.shape,
                )

                new_vials_entry.update(
                    {
                        "vialsDominantLines": vials_dominant_lines,
                        "vialsRecessiveLines": [],  # No recessive lines
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
                # Reuse previous grid - don't update buffer
                logger.debug(
                    f"Frame {self.frame_count}: reusing previous grid (skip update)"
                )
        else:
            logger.debug(f"No valid plate mask in frame #{self.frame_count}")

        # Only update line IDs and cached drawing data when grid was recomputed
        if grid_was_updated:
            # Update best entry from buffer
            if len(self.vials_grid_buffer) > 0:
                current_best = max(self.vials_grid_buffer, key=lambda x: x["confidence"])
                self.best_vials_entry.update(current_best)

            # Update class attributes with tracked line IDs
            self.current_grid_points = self.best_vials_entry.get("points", []) or []

            # Get raw lines from buffer (only dominant lines)
            dominant_lines_raw = self.best_vials_entry.get("vialsDominantLines", [])
            logger.info(f"[DEBUG] Raw dominant lines from buffer: {len(dominant_lines_raw)} lines")
            logger.info(f"[DEBUG] Best entry confidence: {self.best_vials_entry.get('confidence', -1):.3f}")

            # Pass current plate mask for spatial reference
            current_plate_mask = self.prev_plate_mask if self.prev_plate_mask is not None else plate_mask

            # Use line tracker to assign persistent IDs based on spatial position (only dominant lines)
            self.dominant_lines, _ = self.line_tracker.update(
                dominant_lines_raw, [], current_plate_mask
            )
            self.recessive_lines = []  # No recessive lines
            logger.info(f"[DEBUG] After line tracker: {len(self.dominant_lines)} dominant lines")

            # FALLBACK for image processing: If tracker returns no lines but we have raw lines, use them directly
            if len(self.dominant_lines) == 0 and len(dominant_lines_raw) > 0:
                logger.warning(f"[DEBUG] Line tracker returned 0 lines but {len(dominant_lines_raw)} raw lines exist - using raw lines directly")
                self.dominant_lines = [
                    (line[0][0], line[0][1], f"D{i+1}")
                    for i, line in enumerate(dominant_lines_raw)
                ]
                logger.info(f"[DEBUG] Fallback: Created {len(self.dominant_lines)} dominant lines with IDs")

            # Recompute cached drawing data
            self._recompute_cached_drawing_data()
            
            # Run lid state classification if enabled (run every frame for temporal smoothing)
            if self.enable_lid_classification and self.lid_classifier is not None:
                if (self.current_grid_points is not None and 
                    len(self.dominant_lines) > 0 and 
                    len(self.current_grid_points) > 0):
                    # Convert dominant_lines format: (angle_deg, d, line_id) -> ((angle_deg, d), count, line_id)
                    # Use dummy count or count vials per column
                    dominant_lines_for_classifier = []
                    for angle_deg, d, line_id in self.dominant_lines:
                        # Count vials assigned to this column (approximate)
                        # The classifier will do proper assignment, so we use a dummy count
                        dominant_lines_for_classifier.append(((angle_deg, d), 0, line_id))
                    
                    # Convert grid points to centroids (they should already be (x, y) tuples)
                    centroids = [(int(px), int(py)) for px, py in self.current_grid_points]
                    
                    try:
                        self.lid_states = self.lid_classifier.classify(
                            frame, dominant_lines_for_classifier, centroids
                        )
                        if grid_was_updated:
                            logger.debug(f"Lid states classified: {self.lid_states}")
                    except Exception as e:
                        logger.warning(f"Lid classification failed: {e}", exc_info=True)
                        self.lid_states = {}
                else:
                    self.lid_states = {}

    def _recompute_cached_drawing_data(self):
        """Recompute cached line endpoints and plate box from current state."""
        plate_mask = self.prev_plate_mask

        # Cache plate oriented bounding box
        self._cached_plate_box = None
        if plate_mask is not None and self.plate_orientation_angle is not None:
            contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                self._cached_plate_box = box.astype(np.int32)

        # Cache line segment endpoints
        self._cached_line_endpoints = []
        if plate_mask is None or len(self.dominant_lines) == 0:
            return

        x, y, w, h = cv2.boundingRect(plate_mask)
        x2, y2 = x + w, y + h

        for angle_deg, d, line_id in self.dominant_lines:
            angle_rad = math.radians(angle_deg + 90)
            cos_t = math.cos(angle_rad)
            sin_t = math.sin(angle_rad)

            intersections = []
            if abs(cos_t) > 1e-6:
                x_top = (d - sin_t * y) / cos_t
                if x <= x_top <= x2:
                    intersections.append((int(x_top), y))
                x_bottom = (d - sin_t * y2) / cos_t
                if x <= x_bottom <= x2:
                    intersections.append((int(x_bottom), y2))
            if abs(sin_t) > 1e-6:
                y_left = (d - cos_t * x) / sin_t
                if y <= y_left <= y2:
                    intersections.append((x, int(y_left)))
                y_right = (d - cos_t * x2) / sin_t
                if y <= y_right <= y2:
                    intersections.append((x2, int(y_right)))

            if len(intersections) < 2:
                continue

            if len(intersections) == 2:
                p1, p2 = intersections
            else:
                max_dist = 0.0
                p1, p2 = intersections[0], intersections[1]
                for i in range(len(intersections)):
                    for j in range(i + 1, len(intersections)):
                        dist = (intersections[i][0] - intersections[j][0]) ** 2 + (
                            intersections[i][1] - intersections[j][1]
                        ) ** 2
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = intersections[i], intersections[j]

            if p1[1] < p2[1] or (p1[1] == p2[1] and p1[0] < p2[0]):
                text_x, text_y = p1
            else:
                text_x, text_y = p2

            (text_w, text_h), _ = cv2.getTextSize(
                line_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            self._cached_line_endpoints.append((p1, p2, line_id, text_x, text_y, text_w, text_h))

    def draw_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the detected grid lines with IDs on the frame using cached endpoints."""
        annotated = frame.copy()

        # Draw plate oriented bounding box
        if self._cached_plate_box is not None:
            cv2.drawContours(annotated, [self._cached_plate_box], 0, (255, 255, 0), 2)

        # Draw cached line segments with IDs and lid states
        for p1, p2, line_id, text_x, text_y, text_w, text_h in self._cached_line_endpoints:
            cv2.line(annotated, p1, p2, GRID_DOMINANT_COLOR, 2)
            ox, oy = 5, -5
            
            # Draw line ID background and text
            cv2.rectangle(
                annotated,
                (text_x + ox - 2, text_y + oy - text_h - 2),
                (text_x + ox + text_w + 2, text_y + oy + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                annotated, line_id, (text_x + ox, text_y + oy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRID_DOMINANT_COLOR, 1, cv2.LINE_AA,
            )
            
            # Draw lid state if classification is enabled and available
            if self.enable_lid_classification and line_id in self.lid_states:
                lid_state = self.lid_states[line_id]
                # Color: Green for OPEN, Red for CLOSED
                lid_color = (0, 255, 0) if lid_state == "OPEN_LID" else (0, 0, 255)
                lid_text = "OPEN" if lid_state == "OPEN_LID" else "CLOSED"
                
                # Position lid state text below the line ID
                lid_text_y = text_y + oy + text_h + 15
                (lid_w, lid_h), _ = cv2.getTextSize(lid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Draw background for lid state
                cv2.rectangle(
                    annotated,
                    (text_x + ox - 2, lid_text_y - lid_h - 2),
                    (text_x + ox + lid_w + 2, lid_text_y + 2),
                    (0, 0, 0), -1,
                )
                cv2.putText(
                    annotated, lid_text, (text_x + ox, lid_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, lid_color, 1, cv2.LINE_AA,
                )

        # Draw grid points
        if self.current_grid_points is not None:
            for px, py in self.current_grid_points:
                cv2.circle(annotated, (int(px), int(py)), 4, GRID_POINT_COLOR, -1)

        return annotated

    def _draw_line_segments(
        self,
        image: np.ndarray,
        lines_with_data: list,
        plate_mask: np.ndarray | None,
        color: tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw line segments clipped to the plate mask bounding box."""
        if plate_mask is None:
            return

        x, y, w, h = cv2.boundingRect(plate_mask)
        x2, y2 = x + w, y + h

        for idx, item in enumerate(lines_with_data):
            if isinstance(item, tuple) and len(item) == 2:
                line, _ = item
            elif isinstance(item, tuple) and len(item) >= 3:
                line, _, _ = item[:3]
            else:
                continue

            angle_deg, d = line
            angle_rad = math.radians(angle_deg + 90)
            cos_t = math.cos(angle_rad)
            sin_t = math.sin(angle_rad)

            # Find intersections with bounding box edges
            intersections = []

            # Top edge
            if abs(cos_t) > 1e-6:
                x_top = (d - sin_t * y) / cos_t
                if x <= x_top <= x2:
                    intersections.append((int(x_top), y))

            # Bottom edge
            if abs(cos_t) > 1e-6:
                x_bottom = (d - sin_t * y2) / cos_t
                if x <= x_bottom <= x2:
                    intersections.append((int(x_bottom), y2))

            # Left edge
            if abs(sin_t) > 1e-6:
                y_left = (d - cos_t * x) / sin_t
                if y <= y_left <= y2:
                    intersections.append((x, int(y_left)))

            # Right edge
            if abs(sin_t) > 1e-6:
                y_right = (d - cos_t * x2) / sin_t
                if y <= y_right <= y2:
                    intersections.append((x2, int(y_right)))

            if len(intersections) < 2:
                continue

            if len(intersections) == 2:
                p1, p2 = intersections
            else:
                # Pick the two points with maximum distance
                max_dist = 0.0
                p1, p2 = intersections[0], intersections[1]
                for i in range(len(intersections)):
                    for j in range(i + 1, len(intersections)):
                        dist = (intersections[i][0] - intersections[j][0]) ** 2 + (
                            intersections[i][1] - intersections[j][1]
                        ) ** 2
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = intersections[i], intersections[j]

            cv2.line(image, p1, p2, color, thickness)

    def _draw_line_segments_with_ids(
        self,
        image: np.ndarray,
        lines_with_ids: list[tuple[float, float, str]],
        plate_mask: np.ndarray | None,
        color: tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw line segments with IDs clipped to the plate mask bounding box."""
        if plate_mask is None:
            return

        x, y, w, h = cv2.boundingRect(plate_mask)
        x2, y2 = x + w, y + h

        for angle_deg, d, line_id in lines_with_ids:
            angle_rad = math.radians(angle_deg + 90)
            cos_t = math.cos(angle_rad)
            sin_t = math.sin(angle_rad)

            # Find intersections with bounding box edges
            intersections = []

            # Top edge
            if abs(cos_t) > 1e-6:
                x_top = (d - sin_t * y) / cos_t
                if x <= x_top <= x2:
                    intersections.append((int(x_top), y))

            # Bottom edge
            if abs(cos_t) > 1e-6:
                x_bottom = (d - sin_t * y2) / cos_t
                if x <= x_bottom <= x2:
                    intersections.append((int(x_bottom), y2))

            # Left edge
            if abs(sin_t) > 1e-6:
                y_left = (d - cos_t * x) / sin_t
                if y <= y_left <= y2:
                    intersections.append((x, int(y_left)))

            # Right edge
            if abs(sin_t) > 1e-6:
                y_right = (d - cos_t * x2) / sin_t
                if y <= y_right <= y2:
                    intersections.append((x2, int(y_right)))

            if len(intersections) < 2:
                continue

            if len(intersections) == 2:
                p1, p2 = intersections
            else:
                # Pick the two points with maximum distance
                max_dist = 0.0
                p1, p2 = intersections[0], intersections[1]
                for i in range(len(intersections)):
                    for j in range(i + 1, len(intersections)):
                        dist = (intersections[i][0] - intersections[j][0]) ** 2 + (
                            intersections[i][1] - intersections[j][1]
                        ) ** 2
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = intersections[i], intersections[j]

            # Draw the line
            cv2.line(image, p1, p2, color, thickness)
            
            # Determine start point (top-left most point)
            # Sort by y first (topmost), then by x (leftmost)
            if p1[1] < p2[1] or (p1[1] == p2[1] and p1[0] < p2[0]):
                start_x, start_y = p1
            else:
                start_x, start_y = p2
            
            # Add a background for better readability
            (text_w, text_h), baseline = cv2.getTextSize(
                line_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Offset text slightly from the start point for better visibility
            text_offset_x = 5
            text_offset_y = -5
            
            # Draw background rectangle
            cv2.rectangle(
                image,
                (start_x + text_offset_x - 2, start_y + text_offset_y - text_h - 2),
                (start_x + text_offset_x + text_w + 2, start_y + text_offset_y + 2),
                (0, 0, 0),
                -1,
            )
            
            # Draw text
            cv2.putText(
                image,
                line_id,
                (start_x + text_offset_x, start_y + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def get_confidence(self) -> float:
        """Return the current best grid confidence."""
        return self.best_vials_entry.get("confidence", -1.0)
    
    def toggle_debug_display(self):
        """Toggle debug image display on/off."""
        self.show_debug = not self.show_debug
        logger.info(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
        return self.show_debug
    
    def show_debug_images(self, window_name_prefix: str = "Debug", target_size: tuple[int, int] = (320, 240)):
        """
        Display all available debug images in a single composite figure with grid layout.
        
        Args:
            window_name_prefix: Prefix for window name (default: "Debug")
            target_size: Target size (width, height) for each sub-image (default: (320, 240))
        """
        if not self.last_debug_images:
            logger.warning("No debug images available to display")
            return
        
        # Detect which detection method was used based on available keys
        method2_keys = {'wavelet_b_channel', 'wavelet_g_channel', 'wavelet_r_channel', 
                       'segmentation_map', 'thinned_mask', 'connected_components_labels'}
        method5_keys = {'distance_transform', 'step4_distance_transform', 'step5_dist_thresholded',
                       'step6_centrepoints', 'step2_wavelet_coeffs', 'step2b_ca_upsampled'}
        subtraction_keys = {'step1_original_with_plate', 'step2_tray_mask', 'step3_subtraction_raw',
                           'step4a_after_close', 'step4b_after_open'}
        hsv_keys = {'saturation_channel', 'value_channel', 'clear_vial_mask', 'colored_vial_mask'}
        color_based_keys = {'step1_original', 'step2_L', 'step2_A', 'step2_B', 'step3_saturation',
                           'step4_sat_mask', 'step5a_greenness', 'step5b_green_mask', 'step5c_combined'}
        template_matching_edges_keys = {'edges', 'template', 'raw_matches', 'method'}
        template_matching_gradient_keys = {'preprocessed', 'template', 'final_result', 'method'}
        template_matching_lab_chroma_keys = {'preprocessed', 'template', 'final_result', 'method', 'use_edge_gating'}
        log_blob_keys = {'response', 'scale_space', 'method', 'sigma_range', 'use_dog'}
        
        detected_method = None
        # Check method field first for explicit method identification
        method_field = self.last_debug_images.get('method')
        if method_field == 'log_blob_detection':
            detected_method = 'log_blob_detection'
        elif method_field == 'template_matching_lab_chroma':
            detected_method = 'template_matching_lab_chroma'
        elif method_field == 'template_matching_edges':
            detected_method = 'template_matching_edges'
        elif method_field == 'template_matching_gradient':
            detected_method = 'template_matching_gradient'
        elif any(key in self.last_debug_images for key in log_blob_keys):
            detected_method = 'log_blob_detection'
        elif any(key in self.last_debug_images for key in template_matching_lab_chroma_keys):
            # Check if it's LAB chroma by looking for the method or use_edge_gating
            if 'use_edge_gating' in self.last_debug_images or method_field == 'template_matching_lab_chroma':
                detected_method = 'template_matching_lab_chroma'
            elif 'preprocessed' in self.last_debug_images:
                # Could be gradient or LAB chroma - check preprocessed image label or other clues
                detected_method = 'template_matching_lab_chroma'  # Default to LAB chroma if ambiguous
        elif any(key in self.last_debug_images for key in template_matching_edges_keys):
            if 'edges' in self.last_debug_images:
                detected_method = 'template_matching_edges'
            elif 'preprocessed' in self.last_debug_images:
                detected_method = 'template_matching_gradient'
        elif any(key in self.last_debug_images for key in color_based_keys):
            detected_method = 'color_based'
        elif any(key in self.last_debug_images for key in method2_keys):
            detected_method = 'method2'
        elif any(key in self.last_debug_images for key in method5_keys):
            detected_method = 'method5'
        elif any(key in self.last_debug_images for key in subtraction_keys):
            detected_method = 'subtraction'
        elif any(key in self.last_debug_images for key in hsv_keys):
            detected_method = 'hsv'
        
        # Define image categories for each method
        method2_images = [
            ('original_frame', 'Original Frame'),
            ('plate_mask', 'Plate Mask'),
            ('eroded_mask', 'Eroded Mask'),
            ('masked_frame', 'Masked Frame'),
            ('wavelet_b_channel', 'Wavelet B Channel'),
            ('wavelet_g_channel', 'Wavelet G Channel'),
            ('wavelet_r_channel', 'Wavelet R Channel'),
            ('segmentation_map', 'Segmentation Map (K-Means)'),
            ('binary_mask', 'Binary Mask'),
            ('thinned_mask', 'Thinned Mask'),
            ('connected_components_labels', 'Connected Components'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        method5_images = [
            ('step1_grayscale', 'Step 1: Grayscale'),
            ('step2_wavelet_coeffs', 'Step 2: Wavelet Coeffs'),
            ('step2b_ca_upsampled', 'Step 2b: cA Upsampled'),
            ('step3_thresholded', 'Step 3: Thresholded (KEY)'),
            ('step3b_plate_restriction', 'Step 3b: Plate Restrict'),
            ('step4_distance_transform', 'Step 4: Distance Transform'),
            ('step5_dist_thresholded', 'Step 5: Thresholded DT'),
            ('step6_centrepoints', 'Step 6: Centrepoints'),
            ('distance_transform', 'Distance Transform'),
            ('binary_mask', 'Binary Mask'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        subtraction_images = [
            ('step1_original_with_plate', 'Step 1: Original + Plate'),
            ('step2_tray_mask', 'Step 2: Tray Mask'),
            ('step3_subtraction_raw', 'Step 3: Subtraction Raw'),
            ('step4a_after_close', 'Step 4a: After Close'),
            ('step4b_after_open', 'Step 4b: After Open'),
            ('step5_all_components', 'Step 5: All Components'),
            ('step7_filtered_components', 'Step 7: Filtered'),
            ('visualization', 'Final Visualization'),
        ]
        
        hsv_images = [
            ('saturation_channel', 'Saturation Channel'),
            ('value_channel', 'Value Channel'),
            ('clear_vial_mask', 'Clear Vial Mask'),
            ('colored_vial_mask', 'Colored Vial Mask'),
            ('binary_mask', 'Binary Mask'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        color_based_images = [
            ('step1_original', 'Step 1: Original'),
            ('step2_L', 'Step 2: L (Lightness)'),
            ('step2_A', 'Step 2: A (Green-Red)'),
            ('step2_B', 'Step 2: B (Blue-Yellow)'),
            ('step3_saturation', 'Step 3: Saturation (KEY)'),
            ('step4_sat_mask', 'Step 4: Low Sat Mask'),
            ('step5a_greenness', 'Step 5a: Greenness'),
            ('step5b_green_mask', 'Step 5b: Low Green Mask'),
            ('step5c_combined', 'Step 5c: Combined'),
            ('step6_morph', 'Step 6: Morph Clean'),
            ('step7_final_mask', 'Step 7: Final Mask'),
            ('step8_distance', 'Step 8: Distance Transform'),
            ('step9_dist_thresh', 'Step 9: Thresholded DT'),
            ('step10_components', 'Step 10: Centrepoints'),
            ('binary_mask', 'Binary Mask'),
            ('distance_transform', 'Distance Transform'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        template_matching_edges_images = [
            ('edges', 'Step 1: Canny Edges (Domain Adaptation)'),
            ('template_original', 'Step 1.5: Template ROI (Original BGR)'),
            ('template', 'Step 2: Template (Edge Domain)'),
            ('raw_matches', 'Step 3: Raw Template Matches'),
            ('final_result', 'Step 4: Final Result (After NMS)'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        template_matching_gradient_images = [
            ('preprocessed', 'Step 1: Gradient Magnitude (Domain Adaptation)'),
            ('template', 'Step 2: Template (Gradient Domain)'),
            ('final_result', 'Step 3: Final Result'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        template_matching_lab_chroma_images = [
            ('preprocessed', 'Step 1: LAB Chroma (Domain Adaptation)'),
            ('template', 'Step 2: Template (LAB Chroma Domain)'),
            ('final_result', 'Step 3: Final Result (After NMS)'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        log_blob_images = [
            ('response', 'Step 1: LoG/DoG Response (Scale-Space)'),
            ('scale_space', 'Step 2: Scale-Space Pyramid (All Scales)'),
            ('final_result', 'Step 3: Final Result (After NMS + Grid Snap)'),
            ('labeled_image', 'Labeled Image'),
            ('visualization', 'Final Visualization'),
        ]
        
        # Select image categories based on detected method
        if detected_method == 'log_blob_detection':
            image_categories = log_blob_images
        elif detected_method == 'template_matching_lab_chroma':
            image_categories = template_matching_lab_chroma_images
        elif detected_method == 'template_matching_edges':
            image_categories = template_matching_edges_images
        elif detected_method == 'template_matching_gradient':
            image_categories = template_matching_gradient_images
        elif detected_method == 'color_based':
            image_categories = color_based_images
        elif detected_method == 'method2':
            image_categories = method2_images
        elif detected_method == 'method5':
            image_categories = method5_images
        elif detected_method == 'subtraction':
            image_categories = subtraction_images
        elif detected_method == 'hsv':
            image_categories = hsv_images
        else:
            # Fallback: show all available images (for backward compatibility)
            image_categories = (log_blob_images + template_matching_lab_chroma_images + 
                              template_matching_edges_images + template_matching_gradient_images + 
                              color_based_images + method2_images + method5_images + subtraction_images + hsv_images)
        
        # Collect available images
        available_images = []
        for key, label in image_categories:
            if key in self.last_debug_images:
                img = self.last_debug_images[key]
                if img is not None and img.size > 0:
                    # Convert single channel to 3-channel if needed
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    available_images.append((img, label))
        
        # If no images found from predefined categories, try to show all available images
        if not available_images:
            logger.warning("No images found from predefined categories, trying to show all available debug images")
            # Skip non-image values (scalars, strings, dicts, etc.)
            for key, img in self.last_debug_images.items():
                if isinstance(img, np.ndarray) and img is not None and img.size > 0:
                    # Convert single channel to 3-channel if needed
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    available_images.append((img, key))
        
        if not available_images:
            logger.warning("No valid debug images found to display")
            return
        
        # Calculate grid dimensions
        num_images = len(available_images)
        cols = int(np.ceil(np.sqrt(num_images * 1.5)))  # Slightly wider than tall
        rows = int(np.ceil(num_images / cols))
        
        # Resize all images to target size
        resized_images = []
        for img, label in available_images:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append((resized, label))
        
        # Create composite image
        cell_width, cell_height = target_size
        padding = 5
        text_height = 25
        
        composite_width = cols * (cell_width + 2 * padding) + padding
        composite_height = rows * (cell_height + 2 * padding + text_height) + padding
        
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 40  # Dark gray background
        
        # Place images in grid
        for idx, (img, label) in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            
            x_start = padding + col * (cell_width + 2 * padding)
            y_start = padding + row * (cell_height + 2 * padding + text_height)
            
            # Place image
            composite[y_start:y_start + cell_height, x_start:x_start + cell_width] = img
            
            # Add border
            cv2.rectangle(composite, 
                         (x_start - 1, y_start - 1),
                         (x_start + cell_width, y_start + cell_height),
                         (200, 200, 200), 1)
            
            # Add label below image
            label_y = y_start + cell_height + text_height - 5
            label_x = x_start + cell_width // 2
            
            # Get text size for centering
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = x_start + (cell_width - text_w) // 2
            
            # Draw text background
            cv2.rectangle(composite,
                         (label_x - 2, label_y - text_h - 2),
                         (label_x + text_w + 2, label_y + 2),
                         (40, 40, 40), -1)
            
            # Draw text
            cv2.putText(composite, label,
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display composite image
        window_name = f"{window_name_prefix} - All Images ({rows}x{cols})"
        self.debug_window_name = window_name  # Store for closing
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, composite)
        
        logger.info(f"Displaying {num_images} debug images in {rows}x{cols} grid")
    
    def save_debug_images(self, output_folder: str, frame_idx: int | None = None, image_name: str | None = None):
        """
        Save debug images to a folder. For color-based method, saves step_by_step_all.
        For template matching methods, creates and saves a composite image.
        
        Args:
            output_folder: Path to folder where debug image will be saved
            frame_idx: Optional frame index to include in filename (default: None)
            image_name: Optional original image name to use in filename (default: None)
        """
        if not self.last_debug_images:
            logger.warning("No debug images available to save")
            return
        
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            # Check if step_by_step_all exists (color-based method)
            if 'step_by_step_all' in self.last_debug_images:
                step_img = self.last_debug_images['step_by_step_all']
                
                # Check if it's a valid numpy array
                if isinstance(step_img, np.ndarray) and step_img.size > 0:
                    # Convert single channel to 3-channel if needed
                    if len(step_img.shape) == 2:
                        step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
                    elif len(step_img.shape) == 3 and step_img.shape[2] == 4:
                        step_img = cv2.cvtColor(step_img, cv2.COLOR_BGRA2BGR)
                    
                    # Create filename
                    if image_name is not None:
                        filename = f"{image_name}_debug.png"
                    elif frame_idx is not None:
                        filename = f"frame_{frame_idx:06d}_debug.png"
                    else:
                        filename = "debug.png"
                    
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, step_img)
                    logger.info(f"Saved debug image: {filepath}")
                    return
            
            # For template matching methods, create a composite from available images
            # Use the _create_debug_composite method or create a simple composite
            composite = self._create_debug_composite()
            if composite is not None:
                # Create filename
                if image_name is not None:
                    filename = f"{image_name}_debug.png"
                elif frame_idx is not None:
                    filename = f"frame_{frame_idx:06d}_debug.png"
                else:
                    filename = "debug.png"
                
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, composite)
                logger.info(f"Saved debug composite image: {filepath}")
            else:
                # Fallback: save final_result or visualization if available
                if 'final_result' in self.last_debug_images:
                    img = self.last_debug_images['final_result']
                elif 'visualization' in self.last_debug_images:
                    img = self.last_debug_images['visualization']
                else:
                    available_keys = list(self.last_debug_images.keys())[:10]
                    logger.warning(f"No composite image available. Available keys: {available_keys}")
                    return
                
                if isinstance(img, np.ndarray) and img.size > 0:
                    # Convert single channel to 3-channel if needed
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # Create filename
                    if image_name is not None:
                        filename = f"{image_name}_debug.png"
                    elif frame_idx is not None:
                        filename = f"frame_{frame_idx:06d}_debug.png"
                    else:
                        filename = "debug.png"
                    
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, img)
                    logger.info(f"Saved debug image: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")
    
    def _create_debug_composite(self) -> np.ndarray | None:
        """
        Create a composite grid image from all debug images (similar to show_debug_images).
        
        Returns:
            Composite image as numpy array, or None if no images available
        """
        if not self.last_debug_images:
            return None
        
        # Define image categories and their display order (same as show_debug_images)
        image_categories = [
            # LoG/DoG blob detection images (prioritize these)
            ('response', 'Step 1: LoG/DoG Response (Scale-Space)'),
            ('scale_space', 'Step 2: Scale-Space Pyramid (All Scales)'),
            ('final_result', 'Step 3: Final Result (After NMS + Grid Snap)'),
            # LAB chroma template matching images
            ('preprocessed', 'Step 1: LAB Chroma (Domain Adaptation)'),
            ('template', 'Step 2: Template (LAB Chroma Domain)'),
            # Template matching images
            ('edges', 'Step 1: Canny Edges (Domain Adaptation)'),
            ('template', 'Step 2: Template (Edge Domain)'),
            ('raw_matches', 'Step 3: Raw Template Matches'),
            ('final_result', 'Step 4: Final Result (After NMS)'),
            ('preprocessed', 'Step 1: Gradient Magnitude (Domain Adaptation)'),
            # Subtraction method images
            ('step1_original_with_plate', 'Step 1: Original + Plate'),
            ('step2_tray_mask', 'Step 2: Tray Mask'),
            ('step3_subtraction_raw', 'Step 3: Subtraction Raw'),
            ('step4a_after_close', 'Step 4a: After Close'),
            ('step4b_after_open', 'Step 4b: After Open'),
            ('step5_all_components', 'Step 5: All Components'),
            ('step7_filtered_components', 'Step 7: Filtered'),
            ('visualization', 'Final Visualization'),
            ('saturation_channel', 'Saturation Channel'),
            ('value_channel', 'Value Channel'),
            ('clear_vial_mask', 'Clear Vial Mask'),
            ('colored_vial_mask', 'Colored Vial Mask'),
            ('plate_mask', 'Plate Mask'),
            ('tray_mask', 'Tray Mask'),
            ('subtraction_mask_raw', 'Subtraction Raw'),
            ('subtraction_mask_after_close', 'After Close'),
            ('subtraction_mask_after_open', 'After Open'),
            ('subtraction_mask_final', 'Final Mask'),
            ('binary_mask', 'Binary Mask'),
            ('labeled_image', 'Labeled Image'),
            # Wavelet method images
            ('step1_original_gray', 'Step 1: Original Grayscale'),
            ('step2_wavelet_coeffs', 'Step 2: Wavelet Coeffs'),
            ('step2b_ca_upsampled', 'Step 2b: cA Upsampled'),
            ('step3_thresholded', 'Step 3: Thresholded'),
            ('step4a_after_close', 'Step 4a: After Close'),
            ('step4b_after_open', 'Step 4b: After Open'),
            ('step5_plate_mask_final', 'Step 5: Plate Mask Final'),
            ('step6_high_freq', 'Step 6: High-Freq'),
            ('step7_vial_mask_raw', 'Step 7: Vial Mask Raw'),
            ('step8_vial_mask_final', 'Step 8: Vial Mask Final'),
            ('step9_all_contours', 'Step 9: All Contours'),
            ('step10_filtered_contours', 'Step 10: Filtered'),
            ('frequency_analysis', 'Frequency Analysis'),
        ]
        
        # Collect available images
        available_images = []
        for key, label in image_categories:
            if key in self.last_debug_images:
                img = self.last_debug_images[key]
                # Skip non-image values (scalars, strings, dicts, etc.)
                if not isinstance(img, np.ndarray):
                    continue
                if img is not None and img.size > 0:
                    # Convert single channel to 3-channel if needed
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    available_images.append((img, label))
        
        if not available_images:
            return None
        
        # Calculate grid dimensions
        num_images = len(available_images)
        target_size = (320, 240)
        cols = int(np.ceil(np.sqrt(num_images * 1.5)))  # Slightly wider than tall
        rows = int(np.ceil(num_images / cols))
        
        # Resize all images to target size
        resized_images = []
        for img, label in available_images:
            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append((resized, label))
        
        # Create composite image
        cell_width, cell_height = target_size
        padding = 5
        text_height = 25
        
        composite_width = cols * (cell_width + 2 * padding) + padding
        composite_height = rows * (cell_height + 2 * padding + text_height) + padding
        
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 40  # Dark gray background
        
        # Place images in grid
        for idx, (img, label) in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            
            x_start = padding + col * (cell_width + 2 * padding)
            y_start = padding + row * (cell_height + 2 * padding + text_height)
            
            # Place image
            composite[y_start:y_start + cell_height, x_start:x_start + cell_width] = img
            
            # Add border
            cv2.rectangle(composite, 
                         (x_start - 1, y_start - 1),
                         (x_start + cell_width, y_start + cell_height),
                         (200, 200, 200), 1)
            
            # Add label below image
            label_y = y_start + cell_height + text_height - 5
            label_x = x_start + cell_width // 2
            
            # Get text size for centering
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_x = x_start + (cell_width - text_w) // 2
            
            # Draw text background
            cv2.rectangle(composite,
                         (label_x - 2, label_y - text_h - 2),
                         (label_x + text_w + 2, label_y + 2),
                         (40, 40, 40), -1)
            
            # Draw text
            cv2.putText(composite, label,
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1, cv2.LINE_AA)
        
        return composite


def interactive_select_plate_bbox(
    frame: np.ndarray,
    window_name: str = "Select Plate/Rack to Track",
    current_bbox: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int] | None:
    """
    Interactive function to let user draw a plate/rack bounding box on the frame.
    
    Args:
        frame: BGR frame to display
        window_name: Name of the OpenCV window
        current_bbox: Optional current bbox to display (x1, y1, x2, y2)
        
    Returns:
        Selected bounding box as (x1, y1, x2, y2) or None if cancelled
    """
    drawing = False
    start_point = None
    end_point = None
    selected_bbox = current_bbox
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, selected_bbox
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            if start_point and end_point:
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                x2 = max(start_point[0], end_point[0])
                y2 = max(start_point[1], end_point[1])
                if x2 > x1 and y2 > y1:  # Ensure valid bbox
                    selected_bbox = (x1, y1, x2, y2)
    
    display_frame = frame.copy()
    
    # Draw current bbox if provided
    if current_bbox is not None:
        x1, y1, x2, y2 = current_bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Current bbox (green)", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add instructions
    instructions = [
        "Instructions:",
        "1. Click and drag to draw a bounding box around the plate/rack to track",
        "2. Press 'ENTER' or 'SPACE' to confirm selection",
        "3. Press 'ESC' or 'q' to cancel",
        "4. Press 'r' to reset selection"
    ]
    y_offset = 30
    for instruction in instructions:
        cv2.putText(display_frame, instruction, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        temp_frame = display_frame.copy()
        
        # Draw current selection
        if start_point and end_point:
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(temp_frame, "Selected Region", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 or key == 32:  # ENTER or SPACE
            if selected_bbox:
                cv2.destroyWindow(window_name)
                return selected_bbox
            else:
                logger.warning("No bounding box selected. Please draw a box first.")
        
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            cv2.destroyWindow(window_name)
            return None
        
        elif key == ord('r'):  # Reset
            start_point = None
            end_point = None
            selected_bbox = None
            drawing = False


def bbox_to_mask(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> np.ndarray:
    """
    Convert bounding box to binary mask.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
        frame_shape: (height, width) of the frame
        
    Returns:
        Binary mask (uint8) with plate region set to 255
    """
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    return mask




def visualize_video(
    video_path: str,
    annotations_path: str,
    output_path: str | None = None,
    show: bool = True,
    plate_obj_id: str = "4",
    draw_plate_annotation: bool = True,
    start_frame: int = 1000,
    end_frame: int = 2000,
    debug_output_folder: str | None = None,
    enable_lid_classification: bool = True,
    lid_classifier_strip_width: int = 40,
    lid_classifier_buffer_size: int = 5,
    lid_classifier_vial_count_threshold: float = 0.5,
    lid_classifier_reference_state: str = "CLOSED_LID",
) -> None:
    """
    Visualize video with annotations and GridTracker overlay.

    Args:
        video_path: Path to input video
        annotations_path: Path to annotations JSON
        output_path: Optional output video path
        show: Whether to display video window
        plate_obj_id: Object ID for plate in annotations (default "4")
        draw_plate_annotation: Whether to also draw the plate annotation overlay
        start_frame: First frame to process (default 1000)
        end_frame: Last frame to process (default 2000)
        debug_output_folder: Optional folder path to save debug images automatically
    """
    # Convert Windows paths to WSL if needed
    video_path = convert_windows_path_to_wsl(video_path)
    annotations_path = convert_windows_path_to_wsl(annotations_path)
    
    annotations = load_annotations(annotations_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logger.info(f"Starting from frame {start_frame}, ending at frame {end_frame}")

    writer = None
    if output_path is not None:
        # Use H.264 codec for better compression and speed
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
        writer = cv2.VideoWriter(
            output_path, fourcc, fps if fps > 0 else 25, (width, height)
        )
        if not writer.isOpened():
            # Fallback to mp4v if avc1 fails
            logger.warning("H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps if fps > 0 else 25, (width, height)
            )

    # Initialize GridTracker with lid classification
    grid_tracker = GridTrackerWithExternalMask(
        enable_lid_classification=enable_lid_classification,
        lid_classifier_strip_width=lid_classifier_strip_width,
        lid_classifier_buffer_size=lid_classifier_buffer_size,
        lid_classifier_vial_count_threshold=lid_classifier_vial_count_threshold,
        lid_classifier_reference_state=lid_classifier_reference_state,
    )
    
    # Set debug output folder if provided
    if debug_output_folder is not None:
        grid_tracker.debug_output_folder = debug_output_folder
        os.makedirs(debug_output_folder, exist_ok=True)
        logger.info(f"Debug images will be saved to: {debug_output_folder}")

    frame_idx = start_frame
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    avg_fps = 0.0
    
    # Add detailed timing
    timing_stats = {
        'read_frame': [],
        'track_frame': [],
        'draw_grid': [],
        'draw_annotations': [],
        'write_frame': [],
        'display': [],
    }

    while True:
        # Time video read
        t_read = time.perf_counter()
        ret, frame = cap.read()
        timing_stats['read_frame'].append(time.perf_counter() - t_read)
        
        if not ret:
            break
        
        # Stop if we've reached the end frame
        if frame_idx >= end_frame:
            logger.info(f"Reached end frame {end_frame}, stopping")
            break

        frame_key = str(frame_idx)
        frame_ann = annotations.get(frame_key, {})

        # Extract plate mask from annotations
        plate_mask = get_plate_mask_from_annotations(
            frame_ann, (height, width), plate_obj_id
        )

        # Time grid tracking
        t_track = time.perf_counter()
        grid_tracker.track_frame_with_mask(frame, plate_mask)
        timing_stats['track_frame'].append(time.perf_counter() - t_track)

        # Calculate average FPS
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 0:
            avg_fps = fps_frame_count / elapsed_time

        # Time grid drawing
        t_draw_grid = time.perf_counter()
        frame = grid_tracker.draw_grid_overlay(frame)
        timing_stats['draw_grid'].append(time.perf_counter() - t_draw_grid)

        # Time annotation drawing
        t_draw_ann = time.perf_counter()
        skip_ids = set() if draw_plate_annotation else {plate_obj_id}
        if frame_ann:
            draw_frame_polygons(frame, frame_ann, skip_ids=skip_ids)
        timing_stats['draw_annotations'].append(time.perf_counter() - t_draw_ann)

        # Add info overlay
        frame_height = frame.shape[0]
        confidence = grid_tracker.get_confidence()
        
        # Calculate average profiling times
        if len(grid_tracker.profiling_stats['vials_detection_times']) > 0:
            avg_vials = np.mean(grid_tracker.profiling_stats['vials_detection_times']) * 1000
            avg_grid = np.mean(grid_tracker.profiling_stats['grid_generation_times']) * 1000
            avg_total = np.mean(grid_tracker.profiling_stats['total_times']) * 1000
        else:
            avg_vials = avg_grid = avg_total = 0.0
        
        cv2.putText(
            frame,
            f"Avg FPS: {avg_fps:.1f}",
            (10, frame_height - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, frame_height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Grid Conf: {confidence:.3f}",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if confidence > 0.3 else (0, 165, 255),
            2,
        )

        # Add legend
        legend_y = 30
        cv2.putText(
            frame,
            "Grid: Dominant/D# (Red) | Points (Green) | Plate (Cyan)",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        # Add lid classification status
        if grid_tracker.enable_lid_classification and grid_tracker.lid_states:
            open_count = sum(1 for state in grid_tracker.lid_states.values() if state == "OPEN_LID")
            closed_count = sum(1 for state in grid_tracker.lid_states.values() if state == "CLOSED_LID")
            lid_status_text = f"Lids: {open_count} OPEN (Green) | {closed_count} CLOSED (Red)"
            cv2.putText(
                frame,
                lid_status_text,
                (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # Display plate orientation angle
        if grid_tracker.plate_orientation_angle is not None:
            cv2.putText(
                frame,
                f"Plate Angle: {grid_tracker.plate_orientation_angle:.1f}deg",
                (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # Cyan to match plate box
                1,
            )
        
        # Add debug toggle instruction
        debug_status = "ON" if grid_tracker.show_debug else "OFF"
        cv2.putText(
            frame,
            f"Press 'd' to toggle debug images (Current: {debug_status})",
            (10, legend_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255) if grid_tracker.show_debug else (128, 128, 128),
            1,
        )

        # Time video write
        if writer is not None:
            t_write = time.perf_counter()
            writer.write(frame)
            timing_stats['write_frame'].append(time.perf_counter() - t_write)

        # Time display
        if show:
            t_display = time.perf_counter()
            cv2.imshow("WellRack GridTracker Overlay", frame)
            cv2.waitKey(1)
            timing_stats['display'].append(time.perf_counter() - t_display)
            
            # Show debug images if enabled
            if grid_tracker.show_debug:
                grid_tracker.show_debug_images("Debug")
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("d"):  # Toggle debug display
                grid_tracker.toggle_debug_display()
                if not grid_tracker.show_debug:
                    # Close all debug windows
                    try:
                        # Close the composite debug window
                        if grid_tracker.debug_window_name:
                            cv2.destroyWindow(grid_tracker.debug_window_name)
                            grid_tracker.debug_window_name = None
                    except:
                        pass  # Window may not exist
            if key == 32:  # Space to pause
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 in (ord("q"), 32, ord("d")):
                        if key2 == ord("q"):
                            cap.release()
                            if writer is not None:
                                writer.release()
                            cv2.destroyAllWindows()
                            return
                        elif key2 == ord("d"):
                            grid_tracker.toggle_debug_display()
                            if grid_tracker.show_debug:
                                grid_tracker.show_debug_images("Debug")
                            else:
                                # Close the composite debug window
                                try:
                                    cv2.destroyWindow("Debug - All Images")
                                except:
                                    pass  # Window may not exist
                        else:
                            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        logger.info(f"Output video saved to: {output_path}")
    if show:
        cv2.destroyAllWindows()

    frames_processed = frame_idx - start_frame
    logger.info(f"Processed {frames_processed} frames (from {start_frame} to {frame_idx})")
    
    # Print detailed breakdown
    logger.info("=" * 60)
    logger.info("DETAILED TIMING BREAKDOWN")
    logger.info("=" * 60)
    for key, times in timing_stats.items():
        if times:
            avg_ms = np.mean(times) * 1000
            logger.info(f"{key:20s}: {avg_ms:6.2f}ms avg")
    logger.info("=" * 60)
    
    # Print profiling summary
    if len(grid_tracker.profiling_stats['vials_detection_times']) > 0:
        vials_times = grid_tracker.profiling_stats['vials_detection_times']
        grid_times = grid_tracker.profiling_stats['grid_generation_times']
        total_times = grid_tracker.profiling_stats['total_times']
        
        avg_vials = np.mean(vials_times)
        avg_grid = np.mean(grid_times)
        avg_total = np.mean(total_times)
        min_total = np.min(total_times)
        max_total = np.max(total_times)
        
        # Calculate FPS
        fps_avg = 1000.0 / (avg_total * 1000) if avg_total > 0 else 0
        fps_min = 1000.0 / (max_total * 1000) if max_total > 0 else 0
        fps_max = 1000.0 / (min_total * 1000) if min_total > 0 else 0
        fps_vials = 1000.0 / (avg_vials * 1000) if avg_vials > 0 else 0
        fps_grid = 1000.0 / (avg_grid * 1000) if avg_grid > 0 else 0
        
        logger.info("=" * 60)
        logger.info("PROFILING SUMMARY - GridTracker Performance")
        logger.info("=" * 60)
        logger.info(f"Vials Detection:")
        logger.info(f"  - Average: {avg_vials*1000:.2f}ms ({fps_vials:.2f} FPS)")
        logger.info(f"  - Min: {np.min(vials_times)*1000:.2f}ms")
        logger.info(f"  - Max: {np.max(vials_times)*1000:.2f}ms")
        logger.info(f"  - Std Dev: {np.std(vials_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Grid Generation:")
        logger.info(f"  - Average: {avg_grid*1000:.2f}ms ({fps_grid:.2f} FPS)")
        logger.info(f"  - Min: {np.min(grid_times)*1000:.2f}ms")
        logger.info(f"  - Max: {np.max(grid_times)*1000:.2f}ms")
        logger.info(f"  - Std Dev: {np.std(grid_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Total GridTracker:")
        logger.info(f"  - Average: {avg_total*1000:.2f}ms ({fps_avg:.2f} FPS)")
        logger.info(f"  - Min: {min_total*1000:.2f}ms ({fps_max:.2f} FPS)")
        logger.info(f"  - Max: {max_total*1000:.2f}ms ({fps_min:.2f} FPS)")
        logger.info(f"  - Std Dev: {np.std(total_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Percentage breakdown:")
        logger.info(f"  - Vials Detection: {avg_vials/avg_total*100:.1f}%")
        logger.info(f"  - Grid Generation: {avg_grid/avg_total*100:.1f}%")
        logger.info("=" * 60)


def process_images_from_folder(
    input_folder: str,
    output_folder: str | None = None,
    show: bool = True,
    debug_output_folder: str | None = None,
    enable_lid_classification: bool = True,
    lid_classifier_strip_width: int = 40,
    lid_classifier_buffer_size: int = 5,
    lid_classifier_vial_count_threshold: float = 0.5,
    lid_classifier_reference_state: str = "CLOSED_LID",
) -> None:
    """
    Process images from a folder with GridTracker overlay.

    Args:
        input_folder: Path to folder containing images
        output_folder: Optional output folder path (default: input_folder/folder_GridTracker_results)
        show: Whether to display images during processing
        debug_output_folder: Optional folder path to save debug images automatically
    """
    # Convert Windows path to WSL path if needed
    input_folder = convert_windows_path_to_wsl(input_folder)
    if output_folder:
        output_folder = convert_windows_path_to_wsl(output_folder)
    
    input_path = Path(input_folder)
    if not input_path.exists():
        raise RuntimeError(f"Input folder does not exist: {input_folder}")
    
    # Set output folder
    if output_folder is None:
        output_path = input_path / "folder_GridTracker_results_1"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    image_files.sort()
    
    if not image_files:
        raise RuntimeError(f"No image files found in: {input_folder}")
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Collect profiling statistics across all images
    all_vials_times = []
    all_grid_times = []
    all_total_times = []
    
    for idx, image_file in enumerate(image_files):
        # Create a new GridTracker instance for each image to ensure independence
        grid_tracker = GridTrackerWithExternalMask(
            enable_lid_classification=enable_lid_classification,
            lid_classifier_strip_width=lid_classifier_strip_width,
            lid_classifier_buffer_size=lid_classifier_buffer_size,
            lid_classifier_vial_count_threshold=lid_classifier_vial_count_threshold,
            lid_classifier_reference_state=lid_classifier_reference_state,
        )
        
        # Set input folder on detector so it can find ref_images folder
        grid_tracker.detector.input_folder = str(input_path)
        
        # Set debug output folder if provided (save in common folder with image name)
        if debug_output_folder is not None:
            grid_tracker.debug_output_folder = debug_output_folder
            grid_tracker.debug_image_name = image_file.stem
            os.makedirs(debug_output_folder, exist_ok=True)
            logger.info(f"Debug images for {image_file.name} will be saved to: {debug_output_folder}")
        else:
            # Auto-save debug images to default location in output folder
            default_debug_folder = output_path / "debug_images"
            grid_tracker.debug_output_folder = str(default_debug_folder)
            grid_tracker.debug_image_name = image_file.stem
            os.makedirs(default_debug_folder, exist_ok=True)
            logger.info(f"Debug images for {image_file.name} will be saved to: {default_debug_folder}")
        logger.info(f"Processing image {idx + 1}/{len(image_files)}: {image_file.name}")
        
        # Load image
        frame = cv2.imread(str(image_file))
        if frame is None:
            logger.warning(f"Could not load image: {image_file.name}, skipping...")
            continue
        
        height, width = frame.shape[:2]
        
        # Create full-image mask (images are already cropped)
        plate_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Run GridTracker
        grid_tracker.track_frame_with_mask(frame, plate_mask)
        
        # Collect profiling statistics from this image
        if hasattr(grid_tracker.detector, 'last_vials_detection_time') and hasattr(grid_tracker.detector, 'last_grid_generation_time'):
            all_vials_times.append(grid_tracker.detector.last_vials_detection_time)
            all_grid_times.append(grid_tracker.detector.last_grid_generation_time)
            all_total_times.append(
                grid_tracker.detector.last_vials_detection_time + 
                grid_tracker.detector.last_grid_generation_time
            )
        
        # Draw grid overlay
        annotated_frame = grid_tracker.draw_grid_overlay(frame)
        
        # Add info overlay
        frame_height = annotated_frame.shape[0]
        confidence = grid_tracker.get_confidence()
        
        cv2.putText(
            annotated_frame,
            f"Grid Conf: {confidence:.3f}",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if confidence > 0.3 else (0, 165, 255),
            2,
        )
        
        # Save annotated image
        output_file = output_path / f"{image_file.stem}_gridtracker{image_file.suffix}"
        cv2.imwrite(str(output_file), annotated_frame)
        logger.info(f"Saved annotated image to: {output_file}")
        
        if show:
            cv2.imshow("WellRack GridTracker Overlay", annotated_frame)
            
            # Show debug images if enabled
            if grid_tracker.show_debug:
                grid_tracker.show_debug_images("Debug")
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord("q"):
                logger.info("Processing stopped by user")
                break
            elif key == ord("n"):
                # Skip to next image
                continue
            elif key == ord("d"):  # Toggle debug display
                grid_tracker.toggle_debug_display()
                if grid_tracker.show_debug:
                    grid_tracker.show_debug_images("Debug")
                else:
                    # Close all debug windows
                    try:
                        # Close the composite debug window
                        if grid_tracker.debug_window_name:
                            cv2.destroyWindow(grid_tracker.debug_window_name)
                            grid_tracker.debug_window_name = None
                    except:
                        pass  # Window may not exist
                # Continue to show main window
                continue
    
    if show:
        cv2.destroyAllWindows()
    
    logger.info(f"Processed {len(image_files)} images. Results saved to: {output_path}")
    
    # Print profiling summary for all images
    if len(all_vials_times) > 0:
        avg_vials = np.mean(all_vials_times)
        avg_grid = np.mean(all_grid_times)
        avg_total = np.mean(all_total_times)
        min_total = np.min(all_total_times)
        max_total = np.max(all_total_times)
        
        # Calculate FPS
        fps_avg = 1000.0 / (avg_total * 1000) if avg_total > 0 else 0
        fps_min = 1000.0 / (max_total * 1000) if max_total > 0 else 0
        fps_max = 1000.0 / (min_total * 1000) if min_total > 0 else 0
        fps_vials = 1000.0 / (avg_vials * 1000) if avg_vials > 0 else 0
        fps_grid = 1000.0 / (avg_grid * 1000) if avg_grid > 0 else 0
        
        logger.info("=" * 60)
        logger.info("PROFILING SUMMARY - GridTracker Performance (All Images)")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(all_vials_times)}")
        logger.info("")
        logger.info(f"Vials Detection:")
        logger.info(f"  - Average: {avg_vials*1000:.2f}ms ({fps_vials:.2f} FPS)")
        logger.info(f"  - Min: {np.min(all_vials_times)*1000:.2f}ms")
        logger.info(f"  - Max: {np.max(all_vials_times)*1000:.2f}ms")
        logger.info(f"  - Std Dev: {np.std(all_vials_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Grid Generation:")
        logger.info(f"  - Average: {avg_grid*1000:.2f}ms ({fps_grid:.2f} FPS)")
        logger.info(f"  - Min: {np.min(all_grid_times)*1000:.2f}ms")
        logger.info(f"  - Max: {np.max(all_grid_times)*1000:.2f}ms")
        logger.info(f"  - Std Dev: {np.std(all_grid_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Total GridTracker:")
        logger.info(f"  - Average: {avg_total*1000:.2f}ms ({fps_avg:.2f} FPS)")
        logger.info(f"  - Min: {min_total*1000:.2f}ms ({fps_max:.2f} FPS)")
        logger.info(f"  - Max: {max_total*1000:.2f}ms ({fps_min:.2f} FPS)")
        logger.info(f"  - Std Dev: {np.std(all_total_times)*1000:.2f}ms")
        logger.info("")
        logger.info(f"Percentage breakdown:")
        logger.info(f"  - Vials Detection: {avg_vials/avg_total*100:.1f}%")
        logger.info(f"  - Grid Generation: {avg_grid/avg_total*100:.1f}%")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay annotations and WellRack GridTracker on video frames."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="/mnt/c/Users/panos/Panos_Data/Reach_Videos_Data/pipetting/cei_pipetting_cut.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="/mnt/c/Users/panos/Panos_Data/Reach_Videos_Data/Annotations/cei_pipetting/cei_pipetting_cropped_annotations.json",
        help="Path to annotations JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_gridtracker_overlay_10.mp4",
        help="Path to save annotated MP4 (default: output_gridtracker_overlay_new_10.mp4).",
    )
    parser.add_argument(
        "--plate-id",
        type=str,
        default="3",
        help="Object ID for plate in annotations (default: 3).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open OpenCV window.",
    )
    parser.add_argument(
        "--no-plate-overlay",
        action="store_true",
        help="Do not draw the plate annotation overlay (only show grid).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="First frame to process (default: 1000).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=2000,
        help="Last frame to process (default: 2000).",
    )
    parser.add_argument(
        "--debug-output-folder",
        type=str,
        default=None,
        help="Folder path to save debug images automatically (default: None, debug images not saved).",
    )
    parser.add_argument(
        "--no-lid-classification",
        action="store_true",
        help="Disable lid state classification (default: enabled).",
    )
    parser.add_argument(
        "--lid-classifier-strip-width",
        type=int,
        default=40,
        help="Width of vertical column strip for lid classification (default: 40).",
    )
    parser.add_argument(
        "--lid-classifier-buffer-size",
        type=int,
        default=5,
        help="Temporal smoothing buffer size for lid classification (default: 5).",
    )
    parser.add_argument(
        "--lid-classifier-vial-count-threshold",
        type=float,
        default=0.5,
        help="Vial count ratio threshold for structural gate (default: 0.5).",
    )
    parser.add_argument(
        "--lid-classifier-reference-state",
        type=str,
        default="CLOSED_LID",
        choices=["OPEN_LID", "CLOSED_LID"],
        help="Reference state for D1 column (default: CLOSED_LID).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    visualize_video(
        video_path=args.video,
        annotations_path=args.annotations,
        output_path=args.output,
        show=not args.no_show,
        plate_obj_id=args.plate_id,
        draw_plate_annotation=not args.no_plate_overlay,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        debug_output_folder=args.debug_output_folder,
        enable_lid_classification=not args.no_lid_classification,
        lid_classifier_strip_width=args.lid_classifier_strip_width,
        lid_classifier_buffer_size=args.lid_classifier_buffer_size,
        lid_classifier_vial_count_threshold=args.lid_classifier_vial_count_threshold,
        lid_classifier_reference_state=args.lid_classifier_reference_state,
    )


if __name__ == "__main__":
    main()
