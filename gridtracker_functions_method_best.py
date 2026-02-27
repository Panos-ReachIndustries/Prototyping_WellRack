import logging
import math
import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from grid_lines_functions import (
    angle_to_line_params,
    merge_lines_pick_max_support,
    filter_top_percent,
    generate_recessive_angles,
    get_local_dominant_angle_multi_anchors,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


logger = logging.getLogger(__name__)


def get_plate_masks(frame, model_path='yolo_world_l.pt', use_gpu=True, detection_prompts=None):
    try:
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        yolo_model = YOLO(model_path)
        yolo_model.to(device)
        results = yolo_model.predict(frame, classes=[4], conf=0.2)
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            all_masks = results[0].masks.data
            processed_masks = []
            confidence_scores = []
            for i, mask in enumerate(all_masks):
                mask = mask.cpu().numpy()
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=5)
                processed_masks.append(eroded_mask)
                confidence_scores.append(results[0].boxes.conf[i].item())
            return processed_masks, True, confidence_scores
        return [], False, []
    except Exception as e:
        logging.error(f"Error in get_plate_masks: {e}")
        return [], False, []


def compute_grid_confidence(points, inferred_points, filtered_lines, frame_shape):
    """
    Compute confidence score for grid detection.
    Handles None values gracefully when detection fails.
    """
    max_expected_points = 96
    
    # FIX: Handle None values when detection fails
    if points is None:
        points = []
    if inferred_points is None:
        inferred_points = []
    if filtered_lines is None:
        filtered_lines = []
    
    points_ratio = len(points) / max_expected_points if max_expected_points > 0 else 0.0
    inferred_ratio = len(inferred_points) / (len(points) + 1e-6) if len(points) > 0 else 0.0
    inferred_score = 1.0 if inferred_ratio <= 1.0 else (1.0 / inferred_ratio)
    num_lines = len(filtered_lines)
    lines_score = min(num_lines / 20, 1.0)
    confidence = (0.4 * points_ratio) + (0.3 * inferred_score) + (0.3 * lines_score)
    
    # Return 0.0 confidence if no points detected (complete failure)
    if len(points) == 0:
        confidence = 0.0
    
    return confidence


class VialsDetector:
    CANONICAL_RACK_WIDTH = 300

    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.colors = {}
        self.last_mser_time = 0.0
        self.last_grid_time = 0.0
        self.last_tracking_time = 0.0
        self.last_vials_detection_time = 0.0
        self.last_grid_generation_time = 0.0
        self.current_plate_mask = None
        self.plate_orientation_angle = None
        self.reference_orientation_angle = None
        self.next_id = 0

    def extract_plate_orientation(self, mask):
        """Extract dominant orientation angle from plate mask using minAreaRect. Prevents 90° flips using reference angle."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        _, (width, height), angle = rect
        
        if width < height:
            angle = (angle + 90) % 180
        else:
            angle = angle % 180
        
        if self.reference_orientation_angle is not None:
            diff = abs(angle - self.reference_orientation_angle)
            if diff > 90:
                diff = 180 - diff
            
            if 45 <= diff <= 90:
                angle = (angle + 90) % 180
        
        return angle

    def set_plate_mask(self, mask):
        self.current_plate_mask = mask.astype(np.uint8)
        self.plate_orientation_angle = self.extract_plate_orientation(mask)
        if self.plate_orientation_angle is not None and self.reference_orientation_angle is None:
            self.reference_orientation_angle = self.plate_orientation_angle

    def normalize_rack(self, frame):
        """Resize rack image to canonical width, preserving aspect ratio.
        Returns (normalized_frame, scale_factor)."""
        h, w = frame.shape[:2]
        scale = self.CANONICAL_RACK_WIDTH / w
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        normalized = cv2.resize(frame, (self.CANONICAL_RACK_WIDTH, new_h), interpolation=interp)
        return normalized, scale

    def detect_and_track_vials(self, frame, reference_hist=None, frame_count=0):
        try:
            orig_h, orig_w = frame.shape[:2]

            # --- Step 1: Crop to plate mask bounding box ---
            # Instead of normalizing the entire frame, crop to the plate region first
            # so the canonical width applies to the rack, not the full image.
            crop_x, crop_y = 0, 0  # Offset of crop in original frame
            if self.current_plate_mask is not None and np.any(self.current_plate_mask > 0):
                cx, cy, cw, ch = cv2.boundingRect(self.current_plate_mask)
                # Add small padding around the crop (5% of dimensions)
                pad_x = max(10, int(cw * 0.05))
                pad_y = max(10, int(ch * 0.05))
                crop_x = max(0, cx - pad_x)
                crop_y = max(0, cy - pad_y)
                crop_x2 = min(orig_w, cx + cw + pad_x)
                crop_y2 = min(orig_h, cy + ch + pad_y)
                
                cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
                cropped_mask = self.current_plate_mask[crop_y:crop_y2, crop_x:crop_x2]
                crop_h, crop_w = cropped_frame.shape[:2]
                logging.info(
                    f"[CROP] Plate bbox ({cx},{cy},{cw},{ch}) -> crop ({crop_x},{crop_y})-({crop_x2},{crop_y2}) = {crop_w}x{crop_h}"
                )
            else:
                cropped_frame = frame
                cropped_mask = self.current_plate_mask
                crop_h, crop_w = orig_h, orig_w

            # --- Step 2: Normalize the cropped plate region to canonical width ---
            norm_frame, scale = self.normalize_rack(cropped_frame)
            inv_scale = 1.0 / scale
            norm_h, norm_w = norm_frame.shape[:2]
            logging.info(
                f"[NORMALIZE] {crop_w}x{crop_h} -> {norm_w}x{norm_h} (scale={scale:.3f})"
            )

            # Resize the cropped mask to match normalized resolution
            original_mask = self.current_plate_mask
            if cropped_mask is not None:
                self.current_plate_mask = cv2.resize(
                    cropped_mask, (norm_w, norm_h),
                    interpolation=cv2.INTER_NEAREST
                )

            try:
                start_vials_detection = time.perf_counter()
                bboxes, centroids, vis_frame, debug_images = self.detect_vials_4_hsl_minibatch(norm_frame)
                vials_detection_time = time.perf_counter() - start_vials_detection
                self.last_vials_detection_time = vials_detection_time

                start_grid_generation = time.perf_counter()
                final_points, filtered_lines, inferred_points, overlay_grid_lines_frame = self.infer_grid_points(
                    centroids, norm_frame, frame_count
                )
                grid_generation_time = time.perf_counter() - start_grid_generation
                self.last_grid_generation_time = grid_generation_time
            finally:
                self.current_plate_mask = original_mask

            total_time = vials_detection_time + grid_generation_time
            logging.info(
                f"Frame {frame_count}: Vials detection: {vials_detection_time*1000:.2f}ms, "
                f"Grid generation: {grid_generation_time*1000:.2f}ms, "
                f"Total: {total_time*1000:.2f}ms"
            )

            # --- Step 3: Map all results back to original full-frame coordinates ---
            # Two transforms: (1) inv_scale from normalization, (2) +crop_x/y offset

            if centroids is not None:
                centroids = [
                    (int(x * inv_scale) + crop_x, int(y * inv_scale) + crop_y) for x, y in centroids
                ]

            scaled_lines = []
            for line_data in filtered_lines:
                (angle, d), point_count, is_dominant = line_data
                # Scale d back, then offset for the crop origin
                # Line eq: x*cos(t) + y*sin(t) = d
                # After crop offset: (x-crop_x)*cos(t) + (y-crop_y)*sin(t) = d  (in crop coords)
                # In original coords: x*cos(t) + y*sin(t) = d + crop_x*cos(t) + crop_y*sin(t)
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

            if isinstance(bboxes, np.ndarray) and bboxes.ndim >= 2:
                bboxes = cv2.resize(bboxes, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            if debug_images and isinstance(debug_images, dict):
                for key in debug_images:
                    img = debug_images[key]
                    if isinstance(img, np.ndarray) and img.ndim >= 2:
                        interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
                        debug_images[key] = cv2.resize(img, (orig_w, orig_h), interpolation=interp)

            return (
                bboxes,
                bboxes,
                vis_frame,
                debug_images,
                overlay_grid_lines_frame,
                filtered_lines,
                centroids,
                inferred_points,
            )
        except Exception as e:
            logging.error(f"Error occurred in detect_and_track_vials: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], frame.copy(), {}, frame.copy(), [], [], []

    def detect_vials_4_hsl_minibatch(self, frame, min_area=None, max_area=None):
        """HSL filtering + MiniBatchKMeans clustering for vial detection.
        
        IMPROVEMENT: Adaptive area thresholds based on plate size.
        """
        h, w = frame.shape[:2]
        
        # IMPROVEMENT: Calculate adaptive area thresholds from plate dimensions
        if min_area is None or max_area is None:
            if self.current_plate_mask is not None:
                x, y, mask_w, mask_h = cv2.boundingRect(self.current_plate_mask)
                plate_area = mask_w * mask_h
                
                # IMPROVEMENT: More conservative estimation - use actual mask area, not bounding box
                actual_mask_area = np.sum(self.current_plate_mask > 0)
                if actual_mask_area > 0:
                    plate_area = actual_mask_area
                
                # IMPROVEMENT: Estimate vial area - try to detect grid size first
                # For 4-row racks (more common): 4x10 = 40 vials
                # For 8-row racks: 8x12 = 96 vials
                aspect_ratio = mask_w / mask_h if mask_h > 0 else 1.0
                if aspect_ratio < 1.4:
                    expected_vials = 40  # 4-row rack
                elif aspect_ratio < 1.8:
                    expected_vials = 60  # 6-row rack
                else:
                    expected_vials = 96  # 8-row rack
                
                avg_vial_area = plate_area / expected_vials
                
                # IMPROVEMENT: Much more lenient thresholds - use wider range and lower minimum
                # The issue is that after erosion and thinning, vial areas can be much smaller
                # Use 0.1x to 5x range, with absolute minimum of 50 pixels
                min_area = max(50, int(avg_vial_area * 0.1))  # Very lenient: 0.1x, but at least 50px
                max_area = min(20000, int(avg_vial_area * 5.0))  # Very lenient: 5x, higher max
                
                # IMPROVEMENT: Also consider that after erosion, areas shrink significantly
                # Add a fallback minimum that's based on typical small vial size
                min_area = min(min_area, 200)  # Cap minimum at 200 to avoid being too strict
                
                logging.info(f"[DETECTION] Adaptive area thresholds: [{min_area}, {max_area}] (plate: {mask_w}x{mask_h}, actual area: {actual_mask_area}, expected vials: {expected_vials}, avg vial area: {avg_vial_area:.1f})")
            else:
                # IMPROVEMENT: More lenient fallback thresholds
                min_area = 50 if min_area is None else min_area  # Increased from 15 to 50
                max_area = 10000 if max_area is None else max_area  # Increased from 8000
                logging.debug(f"[DETECTION] Using default area thresholds: [{min_area}, {max_area}]")
        
        # Handle mask: create fallback if None or invalid
        if self.current_plate_mask is None:
            logging.warning("[MASK] No mask provided, creating full-frame mask as fallback")
            current_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            current_mask = self.current_plate_mask.copy()
            
            # Check if mask is all zeros or invalid
            mask_area = np.sum(current_mask > 0)
            if mask_area == 0:
                logging.warning("[MASK] Provided mask is all zeros, creating full-frame mask as fallback")
                current_mask = np.ones((h, w), dtype=np.uint8) * 255
            else:
                # Normalize mask format: ensure it's 0/255 (not 0/1)
                if current_mask.max() <= 1:
                    current_mask = (current_mask * 255).astype(np.uint8)
                else:
                    current_mask = current_mask.astype(np.uint8)
                
                # Ensure mask matches frame dimensions
                if current_mask.shape[:2] != (h, w):
                    logging.warning(f"[MASK] Mask shape {current_mask.shape[:2]} doesn't match frame {h}x{w}, resizing")
                    current_mask = cv2.resize(current_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Re-normalize after resize
                    if current_mask.max() <= 1:
                        current_mask = (current_mask * 255).astype(np.uint8)
        
        # Ensure mask is binary (0 or 255) for proper erosion
        current_mask = (current_mask > 127).astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)
        
        # Calculate mask areas for diagnostics
        original_mask_area = np.sum(current_mask > 0)
        
        # Apply erosion - use borderValue=0 so erosion eats inward from image edges
        # (default borderValue for erosion is max, which means a full-frame mask won't erode at all)
        eroded_mask = cv2.erode(current_mask, kernel, iterations=7,
                                borderType=cv2.BORDER_CONSTANT, borderValue=0)
        eroded_mask = eroded_mask.astype(np.uint8)
        eroded_mask_area = np.sum(eroded_mask > 0)
        removed_area = original_mask_area - eroded_mask_area
        removal_percentage = (removed_area / original_mask_area * 100) if original_mask_area > 0 else 0
        
        # Visual debugging: Show erosion effect
        # Masks are already 0/255, just copy for visualization
        original_mask_vis = current_mask.copy()
        eroded_mask_vis = eroded_mask.copy()
        
        # Create side-by-side comparison
        if len(original_mask_vis.shape) == 2:
            original_mask_vis = cv2.cvtColor(original_mask_vis, cv2.COLOR_GRAY2BGR)
        if len(eroded_mask_vis.shape) == 2:
            eroded_mask_vis = cv2.cvtColor(eroded_mask_vis, cv2.COLOR_GRAY2BGR)
        
        # Add text annotations
        cv2.putText(original_mask_vis, f"Original Mask ({original_mask_area}px)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(eroded_mask_vis, f"Eroded Mask ({eroded_mask_area}px, -{removal_percentage:.1f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create overlay showing difference
        diff_mask = original_mask_vis.copy()
        diff_mask[eroded_mask == 0] = [0, 0, 255]  # Red for removed areas
        diff_mask[eroded_mask > 0] = [0, 255, 0]  # Green for kept areas
        cv2.putText(diff_mask, f"Difference (Red=Removed, Green=Kept)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine into single visualization
        vis_h, vis_w = original_mask_vis.shape[:2]
        comparison = np.hstack([original_mask_vis, eroded_mask_vis, diff_mask])
        
        # # Show the comparison
        # cv2.imshow("Mask Erosion Debug: Original | Eroded | Difference", comparison)
        # cv2.waitKey(1)  # Non-blocking, allows other windows to update
        
        # Also show the masked frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=eroded_mask)
        masked_frame_vis = masked_frame.copy()
        # cv2.putText(masked_frame_vis, f"Masked Frame (after erosion)", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.imshow("Masked Frame (After Erosion)", masked_frame_vis)
        # cv2.waitKey(1)
        
        # Log diagnostics
        logging.info(f"[EROSION DEBUG] Original mask: {original_mask_area}px, Eroded: {eroded_mask_area}px, "
                    f"Removed: {removed_area}px ({removal_percentage:.1f}%)")
        if removed_area == 0:
            logging.warning("[EROSION DEBUG] WARNING: Erosion had no effect!")
        elif removal_percentage < 1.0:
            logging.warning(f"[EROSION DEBUG] WARNING: Erosion removed only {removal_percentage:.2f}% - may not be effective")

        hls_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HLS)
        h_channel, l_channel, s_channel = cv2.split(hls_frame)

        h_normalized = h_channel.astype(np.float32) / 180.0
        l_normalized = l_channel.astype(np.float32) / 255.0
        s_normalized = s_channel.astype(np.float32) / 255.0

        # IMPROVEMENT: Adaptive HSL thresholds based on image statistics
        # Calculate adaptive thresholds from masked region
        masked_pixels_l = l_normalized[eroded_mask > 0]
        masked_pixels_s = s_normalized[eroded_mask > 0]
        
        if len(masked_pixels_l) > 0 and len(masked_pixels_s) > 0:
            # Use percentiles for robustness
            l_median = np.median(masked_pixels_l)
            s_median = np.median(masked_pixels_s)
            s_75th = np.percentile(masked_pixels_s, 75)
            
            # IMPROVEMENT: More lenient adaptive thresholds to improve detection rate
            # Reduced lower bound from 0.15 to 0.10, increased upper bound range
            min_lightness = max(0.10, l_median - 0.20)  # More lenient: median - 0.20, but not below 0.10
            max_lightness = min(0.95, l_median + 0.30)  # More lenient: median + 0.30, but not above 0.95
            max_saturation = min(0.6, max(0.25, s_75th * 1.3))  # More lenient: 75th * 1.3, clamped to [0.25, 0.6]
            
            logging.debug(f"[HSL] Adaptive thresholds: L=[{min_lightness:.2f}, {max_lightness:.2f}], S<{max_saturation:.2f}")
        else:
            # IMPROVEMENT: More lenient fallback thresholds
            min_lightness, max_lightness, max_saturation = 0.15, 0.92, 0.5  # More lenient than before
            logging.debug(f"[HSL] Using fixed thresholds (no masked pixels): L=[{min_lightness:.2f}, {max_lightness:.2f}], S<{max_saturation:.2f}")
        
        hsl_mask = ((s_normalized < max_saturation) &
                    (l_normalized >= min_lightness) &
                    (l_normalized <= max_lightness) &
                    (eroded_mask > 0))
        hsl_mask_uint8 = (hsl_mask.astype(np.uint8)) * 255

        feature_vectors = np.column_stack((h_channel.ravel(), l_channel.ravel(), s_channel.ravel()))
        non_zero_indices = np.any(feature_vectors != 0, axis=1)
        n_clusters = 3

        # IMPROVEMENT: Better handling and diagnostics for KMeans clustering
        hsl_mask_coverage = np.sum(hsl_mask_uint8 > 0)
        logging.debug(f"[DETECTION] HSL mask coverage: {hsl_mask_coverage} pixels ({hsl_mask_coverage/(h*w)*100:.1f}% of frame)")

        if np.sum(non_zero_indices) > 0:
            valid_vectors = feature_vectors[non_zero_indices]
            
            # IMPROVEMENT: Check if we have enough valid vectors for clustering
            if len(valid_vectors) < n_clusters:
                logging.warning(f"[DETECTION] Too few valid vectors ({len(valid_vectors)}) for {n_clusters} clusters")
                full_labels = np.full(len(feature_vectors), -1)
            else:
                downsample_factor = 4
                
                if len(valid_vectors) > 500:
                    sampled_vectors = valid_vectors[::downsample_factor]
                else:
                    sampled_vectors = valid_vectors

                try:
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, max_iter=20, batch_size=512, n_init=3)
                    labels_sampled = kmeans.fit_predict(sampled_vectors)

                    if len(sampled_vectors) < len(valid_vectors):
                        indices = np.floor(np.arange(len(valid_vectors)) / downsample_factor).astype(int)
                        indices = np.clip(indices, 0, len(labels_sampled) - 1)
                        labels = labels_sampled[indices]
                    else:
                        labels = labels_sampled

                    full_labels = np.full(len(feature_vectors), -1)
                    full_labels[non_zero_indices] = labels
                except Exception as e:
                    logging.error(f"[DETECTION] KMeans clustering failed: {e}")
                    full_labels = np.full(len(feature_vectors), -1)
        else:
            logging.warning(f"[DETECTION] No non-zero feature vectors found - HSL filtering may be too strict")
            full_labels = np.full(len(feature_vectors), -1)

        seg_map = full_labels.reshape(frame.shape[:2])

        resized_frame_hls = cv2.resize(hls_frame, (seg_map.shape[1], seg_map.shape[0]))
        label_colors = {}
        for label in range(n_clusters):
            label_mask = (seg_map == label)
            if np.any(label_mask):
                label_pixels_hls = resized_frame_hls[label_mask]
                label_colors[label] = (np.mean(label_pixels_hls[:, 0]),
                                      np.mean(label_pixels_hls[:, 1]),
                                      np.mean(label_pixels_hls[:, 2]))

        # IMPROVEMENT: Handle case when no clusters found
        if not label_colors:
            logging.warning(f"[DETECTION] No clusters found in segmentation map - KMeans may have failed")
            # Fallback: use HSL mask directly
            selected_mask = (hsl_mask_uint8 > 0).astype(np.uint8)
        else:
            white_target_score = lambda hls: hls[1] - hls[2]
            closest_label = max(label_colors.items(), key=lambda x: white_target_score(x[1]))[0]
            logging.debug(f"[DETECTION] Selected cluster {closest_label} (white score: {white_target_score(label_colors[closest_label]):.2f})")

            selected_mask = (seg_map == closest_label).astype(np.uint8)
            if selected_mask.shape[:2] != frame.shape[:2]:
                selected_mask = cv2.resize(selected_mask, (frame.shape[1], frame.shape[0]))
            selected_mask = cv2.bitwise_and(selected_mask, hsl_mask_uint8)

        def circular_kernel(radius):
            L = 2*radius + 1
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            kernel = np.zeros((L, L), dtype=np.uint8)
            kernel[x*x + y*y <= radius*radius] = 1
            return kernel

        kernel_circ = circular_kernel(1)
        # IMPROVEMENT: Adaptive erosion - reduce iterations if mask is small
        selected_mask_area = np.sum(selected_mask > 0)
        if self.current_plate_mask is not None:
            plate_area = np.sum(self.current_plate_mask > 0)
            mask_ratio = selected_mask_area / plate_area if plate_area > 0 else 0
            # If mask covers less than 10% of plate, use fewer erosion iterations
            erosion_iterations = 3 if mask_ratio < 0.10 else 5
        else:
            erosion_iterations = 5
        
        thinned_mask = cv2.erode(selected_mask.astype(np.uint8), kernel_circ, iterations=erosion_iterations) * 255
        logging.debug(f"[DETECTION] Erosion: {erosion_iterations} iterations, selected_mask area: {selected_mask_area}px")

        num_labels, labels_full, stats, centroids = cv2.connectedComponentsWithStats(
            thinned_mask, connectivity=8
        )

        # IMPROVEMENT: Two-pass approach - first collect all areas, then adapt thresholds if needed
        all_areas = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            all_areas.append(area)
        
        # IMPROVEMENT: Dynamically adjust thresholds based on detected component areas
        # If initial thresholds reject all components, use component area statistics instead
        if len(all_areas) > 0:
            area_median = np.median(all_areas)
            area_75th = np.percentile(all_areas, 75)
            area_25th = np.percentile(all_areas, 25)
            
            # Check if current thresholds would reject everything
            components_in_range = sum(1 for a in all_areas if min_area < a < max_area)
            
            if components_in_range == 0 and len(all_areas) > 0:
                # IMPROVEMENT: Adapt thresholds based on actual component areas
                # Use percentiles of detected areas as new thresholds
                adaptive_min = max(30, int(area_25th * 0.5))  # 50% of 25th percentile, but at least 30
                adaptive_max = min(20000, int(area_75th * 3.0))  # 3x of 75th percentile
                
                logging.warning(f"[DETECTION] Initial thresholds [{min_area}, {max_area}] rejected all {len(all_areas)} components")
                logging.warning(f"[DETECTION] Component area stats: min={min(all_areas):.1f}, 25th={area_25th:.1f}, median={area_median:.1f}, 75th={area_75th:.1f}, max={max(all_areas):.1f}")
                logging.warning(f"[DETECTION] Adapting thresholds to [{adaptive_min}, {adaptive_max}] based on component statistics")
                
                min_area = adaptive_min
                max_area = adaptive_max

        # IMPROVEMENT: Better diagnostics for detection failures
        valid_indices = []
        rejected_by_area = []
        rejected_by_mask = []
        
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            cx, cy = centroids[label_idx]
            
            # Check area first
            if not (min_area < area < max_area):
                rejected_by_area.append((label_idx, area, min_area, max_area))
                continue
            
            # Check if centroid is within eroded mask
            if (0 <= int(cx) < eroded_mask.shape[1] and
                0 <= int(cy) < eroded_mask.shape[0] and
                eroded_mask[int(cy), int(cx)] > 0):
                valid_indices.append(label_idx)
            else:
                rejected_by_mask.append((label_idx, (cx, cy)))

        valid_indices = np.array(valid_indices, dtype=int)
        
        # Log diagnostics if no vials detected
        if len(valid_indices) == 0:
            logging.warning(f"[DETECTION] No vials detected! Diagnostics:")
            logging.warning(f"  - Total components found: {num_labels - 1}")
            if all_areas:
                logging.warning(f"  - Component areas: min={min(all_areas):.1f}, max={max(all_areas):.1f}, median={np.median(all_areas):.1f}")
                logging.warning(f"  - Area thresholds: [{min_area}, {max_area}]")
            if rejected_by_area:
                logging.warning(f"  - Rejected by area: {len(rejected_by_area)} components")
                sample = rejected_by_area[:3]
                for idx, area, min_a, max_a in sample:
                    logging.warning(f"    Component {idx}: area={area:.1f} (threshold: [{min_a}, {max_a}])")
            if rejected_by_mask:
                logging.warning(f"  - Rejected by mask: {len(rejected_by_mask)} components")
            if num_labels == 1:
                logging.warning(f"  - No components found in thinned_mask - HSL/KMeans may have failed")
                logging.warning(f"  - Check: hsl_mask coverage, KMeans clustering, selected_mask")

        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_indices, start=1)}
        vial_labels_remapped = np.zeros_like(labels_full, dtype=np.uint8)
        for old_label, new_label in label_mapping.items():
            vial_labels_remapped[labels_full == old_label] = new_label

        vial_centroids_filtered = centroids[valid_indices] if len(valid_indices) > 0 else None
        if vial_centroids_filtered is not None:
            vial_centroids_filtered = [(int(x), int(y)) for x, y in vial_centroids_filtered]
            logging.info(f"[DETECTION] Successfully detected {len(vial_centroids_filtered)} vials")
        else:
            logging.warning(f"[DETECTION] No vials detected - returning None centroids")

        overlay = frame.copy()
        n_labels_vis = vial_labels_remapped.max() + 1
        colors = np.zeros((n_labels_vis, 3), dtype=np.uint8)
        colors[1:] = np.random.randint(50, 255, size=(n_labels_vis - 1, 3), dtype=np.uint8)

        for label_num in range(1, n_labels_vis):
            overlay[vial_labels_remapped == label_num] = colors[label_num]

        vis_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if vial_centroids_filtered is not None:
            for cx, cy in vial_centroids_filtered:
                cv2.circle(vis_frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(vis_frame, f"Method 4 (HSL+MiniBatch): {len(vial_centroids_filtered) if vial_centroids_filtered else 0} vials",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        seg_map_display = cv2.normalize(seg_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if seg_map_display.shape[:2] != frame.shape[:2]:
            seg_map_display = cv2.resize(seg_map_display, (frame.shape[1], frame.shape[0]))

        debug_dict = {
            'original_frame': frame.copy(),
            'plate_mask': (current_mask * 255).astype(np.uint8),
            'eroded_mask': cv2.bitwise_and(frame, frame, mask=eroded_mask),
            'masked_frame': masked_frame,
            'hls_h_channel': h_channel,
            'hls_l_channel': l_channel,
            'hls_s_channel': s_channel,
            'hsl_filter_mask': hsl_mask_uint8,
            'segmentation_map': seg_map_display,
            'binary_mask': (selected_mask * 255).astype(np.uint8),
            'thinned_mask': thinned_mask,
            'connected_components_labels': (labels_full.astype(np.uint8) * (255 // max(1, num_labels - 1))).astype(np.uint8),
            'labeled_image': (vial_labels_remapped.astype(np.uint8) * 50),
            'visualization': vis_frame,
        }

        return vial_labels_remapped, vial_centroids_filtered, vis_frame, debug_dict

    def infer_grid_points(self, points, frame, frame_count=0):
        start_grid = time.time()
        # FIX: Handle None centroids (when vial detection fails)
        if points is None:
            logging.warning("[GRID] No points provided (vial detection returned None)")
            self.last_grid_time = time.time() - start_grid
            return [], [], [], frame
        if len(points) < 3:
            logging.debug(f"[GRID] Too few points ({len(points)} < 3) for grid inference")
            self.last_grid_time = time.time() - start_grid
            return points, [], [], frame
        overlay_frame = frame.copy()
        for px, py in points:
            cv2.circle(overlay_frame, (int(px), int(py)), 6, (0, 0, 255), -1)
        
        plate_longest_axis_angle = None
        if self.plate_orientation_angle is not None:
            plate_longest_axis_angle = self.plate_orientation_angle
        elif self.current_plate_mask is not None:
            plate_longest_axis_angle = self.extract_plate_orientation(self.current_plate_mask)
        
        if self.plate_orientation_angle is not None:
            reference_angle = self.plate_orientation_angle
        else:
            computed_angle = get_local_dominant_angle_multi_anchors(
                points, 
                num_anchors=min(5, len(points)),
                bin_width=6, 
                distance_threshold=100.0, 
                outlier_threshold=5.0, 
                debug=False
            )
            
            if plate_longest_axis_angle is not None:
                angle_diff = abs(computed_angle - plate_longest_axis_angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                max_allowed_diff = 30.0
                if angle_diff > max_allowed_diff:
                    reference_angle = plate_longest_axis_angle
                else:
                    reference_angle = computed_angle
            else:
                reference_angle = computed_angle
        
        filtered_dominant_angles = [
            (points[i][0], points[i][1], reference_angle)
            for i in range(len(points))
        ]
        
        dominant_lines_raw = []
        for x, y, dom_angle in filtered_dominant_angles:
            line_angle, line_d = angle_to_line_params(x, y, dom_angle)
            dominant_lines_raw.append(((line_angle, line_d), dom_angle))
        
        recessive_lines_raw = generate_recessive_angles(filtered_dominant_angles)
        
        merged_dom_lines = merge_lines_pick_max_support(
            dominant_lines_raw, angle_threshold=30.0, distance_threshold=30.0
        )
        
        merged_rec_lines = merge_lines_pick_max_support(
            recessive_lines_raw, angle_threshold=30.0, distance_threshold=30.0
        )
        
        # IMPROVEMENT: Adaptive percentile filtering with better grid size detection
        # Try to estimate expected lines from plate dimensions
        num_merged_dom = len(merged_dom_lines)
        num_merged_rec = len(merged_rec_lines)
        
        # Estimate expected lines from plate (same logic as spacing calculation)
        if self.current_plate_mask is not None:
            x, y, w, h = cv2.boundingRect(self.current_plate_mask)
            plate_width = max(w, h)
            plate_height = min(w, h)
            aspect_ratio = plate_width / plate_height if plate_height > 0 else 1.0
            
            if aspect_ratio < 1.4:
                expected_dom = 4  # 4-row rack
                expected_rec = 10
            elif aspect_ratio < 1.8:
                expected_dom = 6
                expected_rec = 11
            else:
                expected_dom = 8  # 8-row rack
                expected_rec = 12
        else:
            # Fallback: use conservative defaults
            expected_dom = 4  # Default to 4-row (more common in dataset)
            expected_rec = 10
        
        # IMPROVEMENT: More lenient filtering - only filter aggressively if way too many lines
        # Base percentile increased from 15% to 20% for better detection rate
        base_percent = 20
        
        if num_merged_dom > expected_dom * 2.0:  # Only filter if >2x expected
            # Too many lines: use more aggressive filtering
            dom_percent = max(12, base_percent - (num_merged_dom - expected_dom) // 3)
        else:
            dom_percent = base_percent  # Use base percentile (more lenient)
        
        if num_merged_rec > expected_rec * 2.0:
            rec_percent = max(12, base_percent - (num_merged_rec - expected_rec) // 3)
        else:
            rec_percent = base_percent
        
        # IMPROVEMENT: Ensure we keep at least 2-3 lines even if filtering is aggressive
        # This prevents complete detection failure
        min_lines_to_keep = max(2, expected_dom // 2)
        if num_merged_dom > 0:
            estimated_kept = int(num_merged_dom * dom_percent / 100)
            if estimated_kept < min_lines_to_keep and num_merged_dom >= min_lines_to_keep:
                # Adjust percentile to keep minimum lines
                dom_percent = min(100, int((min_lines_to_keep / num_merged_dom) * 100) + 5)
        
        logging.info(f"[GRID] Filtering: {num_merged_dom} dom lines (expected: {expected_dom}, percent={dom_percent}%), {num_merged_rec} rec lines (expected: {expected_rec}, percent={rec_percent}%)")
        
        final_dom_filtered_lines = filter_top_percent(merged_dom_lines, percent=dom_percent)
        final_rec_filtered_lines = filter_top_percent(merged_rec_lines, percent=rec_percent)
        
        # Log if filtering removed too many lines
        if len(final_dom_filtered_lines) < min_lines_to_keep and num_merged_dom >= min_lines_to_keep:
            logging.warning(f"[GRID] WARNING: Filtering removed too many lines ({len(final_dom_filtered_lines)} < {min_lines_to_keep} minimum)")
        
        dominant_lines = []
        for (line_params, _) in final_dom_filtered_lines:
            dominant_lines.append((line_params, 1))
        
        recessive_lines = []
        for (line_params, _) in final_rec_filtered_lines:
            recessive_lines.append((line_params, 1))

        # IMPROVEMENT: Calculate min_line_spacing from plate dimensions with adaptive grid size detection
        # This provides consistent spacing regardless of detection quality
        if self.current_plate_mask is not None:
            # Get plate bounding box
            x, y, w, h = cv2.boundingRect(self.current_plate_mask)
            plate_width = max(w, h)  # Use longer dimension
            plate_height = min(w, h)
            
            # IMPROVEMENT: Adaptive grid size detection based on plate aspect ratio
            aspect_ratio = plate_width / plate_height if plate_height > 0 else 1.0
            
            # Estimate grid size: 4-row racks are more square, 8-row racks are more rectangular
            if aspect_ratio < 1.4:
                estimated_rows = 4  # 4-row rack
                expected_cols = 10
            elif aspect_ratio < 1.8:
                estimated_rows = 6  # Could be either
                expected_cols = 11
            else:
                estimated_rows = 8  # 8-row rack
                expected_cols = 12
            
            # Also try to estimate from detected points if available
            if len(points) >= 4:
                points_array = np.array(points, dtype=np.float32)
                if self.plate_orientation_angle is not None:
                    # Project points onto dominant direction to estimate rows
                    angle_rad = np.radians(self.plate_orientation_angle + 90)
                    cos_t = np.cos(angle_rad)
                    sin_t = np.sin(angle_rad)
                    projections = points_array[:, 0] * cos_t + points_array[:, 1] * sin_t
                    
                    proj_range = np.max(projections) - np.min(projections)
                    if proj_range > 0 and len(projections) > 1:
                        sorted_proj = np.sort(projections)
                        avg_spacing = np.median(np.diff(sorted_proj))
                        if avg_spacing > 0:
                            point_based_rows = max(4, min(8, int(round(proj_range / avg_spacing))))
                            # Use point-based estimate if it's reasonable
                            if 3 <= point_based_rows <= 8:
                                estimated_rows = point_based_rows
                                logging.debug(f"[GRID] Point-based row estimate: {point_based_rows}")
            
            logging.info(f"[GRID] Plate dimensions: {w}x{h} (aspect: {aspect_ratio:.2f}), estimated grid: {estimated_rows} rows × {expected_cols} cols")
            
            # Calculate expected spacing based on detected grid size
            if plate_height > 0:
                expected_row_spacing = plate_height / (estimated_rows + 1)
                expected_col_spacing = plate_width / (expected_cols + 1)
                
                # Use the smaller spacing as min_line_spacing
                # IMPROVEMENT: More lenient tolerance (0.6 instead of 0.8) to allow closer lines
                min_line_spacing = min(expected_row_spacing, expected_col_spacing) * 0.6
                
                # Ensure minimum threshold (fallback for very small plates)
                min_line_spacing = max(min_line_spacing, 10.0)  # Reduced from 15.0
                
                logging.info(f"[GRID] Expected spacing: row={expected_row_spacing:.1f}px, col={expected_col_spacing:.1f}px, min_line_spacing={min_line_spacing:.1f}px")
            else:
                min_line_spacing = 20.0  # Reduced from 30.0
        elif len(points) >= 2:
            # Fallback: use detected points if no plate mask available
            points_array = np.array(points, dtype=np.float32)
            distances = cdist(points_array, points_array, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            nearest_neighbor_dists = np.min(distances, axis=1)
            avg_vial_spacing = np.median(nearest_neighbor_dists)
            avg_vial_radius = avg_vial_spacing / 2.0
            min_line_spacing = 2.0 * avg_vial_radius
            logging.info(f"[GRID] Using point-based spacing: {min_line_spacing:.1f}px (fallback)")
        else:
            min_line_spacing = 30.0

        if len(dominant_lines) > 1:
            dominant_line_distances = []
            for (line_params, point_count) in dominant_lines:
                angle, d = line_params
                dominant_line_distances.append(((angle, d), point_count, d))

            dominant_line_distances.sort(key=lambda x: x[2])

            filtered_dominant_lines = []
            for i, (line_params, point_count, d) in enumerate(dominant_line_distances):
                keep_line = True
                for (kept_params, _, kept_d) in filtered_dominant_lines:
                    if abs(d - kept_d) < min_line_spacing:
                        keep_line = False
                        logging.debug(f"[GRID] Filtering line at d={d:.1f}px (too close to d={kept_d:.1f}px, spacing={abs(d-kept_d):.1f}px < {min_line_spacing:.1f}px)")
                        break

                if keep_line:
                    filtered_dominant_lines.append((line_params, point_count, d))

            dominant_lines = [(line_params, point_count) for (line_params, point_count, _) in filtered_dominant_lines]
            logging.info(f"[GRID] Proximity filtering: {len(dominant_line_distances)} -> {len(dominant_lines)} dominant lines")

        final_filtered_lines = []
        for line_params, point_count in dominant_lines:
            final_filtered_lines.append((line_params, point_count, True))
        for line_params, point_count in recessive_lines:
            final_filtered_lines.append((line_params, point_count, False))
        
        self.last_grid_time = time.time() - start_grid
        return [], final_filtered_lines, [], overlay_frame
