"""
Deterministic lid state classifier using reference-based subtraction imaging.

This module implements a reference-based classifier that determines whether
each dominant column (D1...D8) has open or closed lids by comparing column
image strips against D1 (reference, assumed OPEN).
"""

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LidStateClassifier:
    """
    Reference-based lid state classifier using subtraction imaging with two-gate system.
    
    Algorithm:
    1. Crop tight vertical strips for each column (D1, D2, ..., D8)
    2. Use D1 as reference (configurable: OPEN or CLOSED)
    3. Compute image difference: |D1 - Di| for each column
    4. Gate 1 (Structural - Relative Ratio Check):
       - Check: vials/ref_vials ≤ vial_count_threshold → OPEN_LID
       - Example: If threshold=0.2, columns with ≤20% of reference vials → OPEN
    5. Gate 2 (Visual): Threshold difference to classify: similar → same state, different → opposite state
    6. Apply temporal smoothing (5-frame majority vote)
    """

    def __init__(
        self,
        strip_width: int = 40,
        buffer_size: int = 5,
        vial_count_threshold: float = 0.5,  # Ratio threshold for structural gate
        diff_threshold: Optional[float] = None,  # Visual similarity threshold (auto if None)
        diff_threshold_method: str = "percentile",  # "percentile", "median_iqr", or "fixed"
        iqr_multiplier: float = 1.0,  # Multiplier for IQR in median_iqr method (default: 1.0, more conservative than 0.5)
        min_visual_threshold: float = 0.0,  # Floor for adaptive visual threshold; prevents FPs when adaptive threshold is too low
        reference_state: str = "CLOSED_LID",
        use_fft_features: bool = False,  # Fix B: use FFT texture power instead of raw MAD (opt-in)
        use_pairwise_clustering: bool = False,  # Fix C: K-Means on FFT features, no D1 assumption (opt-in)
        min_feature_separation: float = 0.08,  # Min gap between adjacent sorted features to split
        profile_sigma_factor: float = 0.0,  # Gaussian weighting for 1D profile (0 = box/mean, 0.20 recommended)
        debug: bool = False,
    ):
        """
        Initialize the LidStateClassifier with two-gate system.

        Args:
            strip_width: Width of vertical column strip to extract (pixels)
            buffer_size: Number of frames to buffer for temporal smoothing
            vial_count_threshold: Vial ratio threshold for Gate 1 (structural gate)
                                 If vial_count / reference_vials ≤ threshold → OPEN_LID
                                 Example: threshold=0.2 means columns with ≤20% of reference vials → OPEN
            diff_threshold: Fixed difference threshold for Gate 2 (visual gate)
                           If None, computed adaptively using diff_threshold_method
            diff_threshold_method: Method for adaptive threshold:
                                  - "percentile": 60th percentile of differences
                                  - "median_iqr": median + iqr_multiplier * IQR
                                  - "fixed": use diff_threshold value
            iqr_multiplier: Multiplier for IQR when using median_iqr method (default: 1.0)
                           Higher values = more conservative (fewer OPEN classifications)
                           Lower values = more aggressive (more OPEN classifications)
            min_visual_threshold: Minimum floor for the adaptive visual threshold (default: 0.0 = disabled)
                                 When adaptive threshold < this floor, the floor is used instead.
                                 Prevents OPEN misclassification when per-image threshold collapses too low
                                 (e.g. with N=2 columns, median+IQR equals max(diffs) → always triggers OPEN)
            reference_state: State of D1 reference ("OPEN_LID" or "CLOSED_LID")
            profile_sigma_factor: Gaussian weighting width for 1D profile computation, as a fraction of
                                  strip_width (default 0.0 = plain mean; 0.20 recommended for racks where
                                  the leftmost column strip may include the rack's internal cell-wall
                                  divider, which contaminates the FFT with spurious high-frequency content).
                                  sigma = strip_width * profile_sigma_factor in pixels.
            debug: Enable debug mode for visualization
        """
        self.strip_width = strip_width
        self.buffer_size = buffer_size
        self.vial_count_threshold = vial_count_threshold
        self.diff_threshold = diff_threshold
        self.diff_threshold_method = diff_threshold_method
        self.iqr_multiplier = iqr_multiplier
        self.min_visual_threshold = min_visual_threshold
        self.reference_state = reference_state
        self.use_fft_features = use_fft_features
        self.use_pairwise_clustering = use_pairwise_clustering
        self.min_feature_separation = min_feature_separation
        self.profile_sigma_factor = profile_sigma_factor
        self.debug = debug

        # CLAHE for illumination normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Temporal smoothing buffers per column
        self.column_buffers: Dict[str, deque] = {}

        # Debug storage
        self.debug_info: Dict[str, Dict] = {}
        
        # Gateway interaction storage (for JSON export)
        self.gateway_info: List[Dict] = []

        logger.info(
            f"LidStateClassifier initialized (two-gate system): strip_width={strip_width}, "
            f"buffer_size={buffer_size}, vial_count_threshold={vial_count_threshold}, "
            f"diff_threshold={diff_threshold}, diff_threshold_method={diff_threshold_method}, "
            f"iqr_multiplier={iqr_multiplier}, min_visual_threshold={min_visual_threshold}, "
            f"reference_state={reference_state}, use_fft_features={use_fft_features}, "
            f"use_pairwise_clustering={use_pairwise_clustering}, "
            f"min_feature_separation={min_feature_separation}, "
            f"profile_sigma_factor={profile_sigma_factor}, debug={debug}"
        )

    def _normal_form_distance(
        self, point: Tuple[float, float], angle_deg: float, d: float
    ) -> float:
        """
        Compute normal form distance from a point to a line.

        Args:
            point: (x, y) coordinates
            angle_deg: Angle of the line in degrees
            d: Distance parameter

        Returns:
            Distance from point to line
        """
        x, y = point
        angle_rad = math.radians(angle_deg + 90)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        distance = abs(cos_theta * x + sin_theta * y - d) / math.sqrt(
            cos_theta**2 + sin_theta**2
        )
        return distance

    def assign_vials_to_columns(
        self,
        dominant_lines: List[Tuple],
        centroids: List[Tuple[int, int]],
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Assign each vial centroid to its nearest dominant line (column).

        Args:
            dominant_lines: List of ((angle_deg, d), count, line_id) or ((angle_deg, d), count)
            centroids: List of (x, y) vial centroids

        Returns:
            Dictionary: {column_id: [(x1, y1), (x2, y2), ...]}
        """
        column_vials: Dict[str, List[Tuple[int, int]]] = {}

        for centroid in centroids:
            min_distance = float("inf")
            closest_column_id = None

            for line_data in dominant_lines:
                if len(line_data) >= 3:
                    (angle_deg, d), count, line_id = line_data
                elif len(line_data) >= 2:
                    (angle_deg, d), count = line_data[:2]
                    line_id = f"D{len(column_vials) + 1}"
                else:
                    continue

                distance = self._normal_form_distance(centroid, angle_deg, d)

                if distance < min_distance:
                    min_distance = distance
                    closest_column_id = line_id

            if closest_column_id:
                if closest_column_id not in column_vials:
                    column_vials[closest_column_id] = []
                column_vials[closest_column_id].append(centroid)

        return column_vials

    def extract_column_strip(
        self,
        frame: np.ndarray,
        vials: List[Tuple[int, int]],
        angle_deg: float,
        d: float,
    ) -> Optional[np.ndarray]:
        """
        Extract a TIGHT vertical strip containing all vials in a column.

        Args:
            frame: Input image (BGR)
            vials: List of (x, y) centroids for vials in this column
            angle_deg: Column line angle
            d: Column line distance parameter

        Returns:
            Grayscale column strip (normalized), or None if extraction fails
        """
        if not vials:
            return None

        h, w = frame.shape[:2]
        
        # Sort vials by y-coordinate (top to bottom)
        vials_sorted = sorted(vials, key=lambda v: v[1])
        
        # Compute line position at top and bottom of image
        angle_rad = math.radians(angle_deg + 90)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        # For vertical-ish lines (angle near 0 or 180), x = d / cos(theta)
        # Get x-coordinate of line at different y values
        if abs(cos_theta) > 0.1:  # Vertical-ish line
            # VERY TIGHT bounding box: minimal padding
            y_min = max(0, min(v[1] for v in vials) - 5)  # Very minimal padding
            y_max = min(h, max(v[1] for v in vials) + 5)  # Very minimal padding
            
            # Compute x-center from line equation
            y_mid = (y_min + y_max) / 2
            x_center = int((d - sin_theta * y_mid) / cos_theta)
            
            # TIGHTER strip centered on line (use actual strip_width, default 50)
            half_width = self.strip_width // 2
            x1 = max(0, x_center - half_width)
            x2 = min(w, x_center + half_width)
            
            strip = frame[int(y_min):int(y_max), x1:x2]
            
        else:  # Horizontal-ish line
            # VERY TIGHT bounding box for horizontal
            x_min = max(0, min(v[0] for v in vials) - 5)  # Very minimal padding
            x_max = min(w, max(v[0] for v in vials) + 5)  # Very minimal padding
            
            y_mid = (max(v[1] for v in vials) + min(v[1] for v in vials)) / 2
            y_center = int(y_mid)
            
            half_width = self.strip_width // 2
            y1 = max(0, y_center - half_width)
            y2 = min(h, y_center + half_width)
            
            strip = frame[y1:y2, int(x_min):int(x_max)]

        if strip.size == 0:
            return None

        # Convert to grayscale and normalize
        if len(strip.shape) == 3:
            strip_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        else:
            strip_gray = strip

        # Apply CLAHE for illumination normalization
        strip_normalized = self.clahe.apply(strip_gray)

        return strip_normalized

    def compute_difference(
        self, reference: np.ndarray, target: np.ndarray
    ) -> float:
        """
        Compute mean absolute difference between reference and target strips.

        Args:
            reference: Reference column strip (D1, OPEN)
            target: Target column strip (Di)

        Returns:
            Mean absolute difference (normalized to [0, 255])
        """
        # Resize target to match reference dimensions
        if reference.shape != target.shape:
            target_resized = cv2.resize(
                target, (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            target_resized = target

        # Compute absolute difference
        diff = np.abs(reference.astype(np.float32) - target_resized.astype(np.float32))
        mean_diff = np.mean(diff)

        return float(mean_diff)

    def _compute_fft_texture_feature(self, strip: np.ndarray) -> float:
        """
        Compute FFT-based texture feature for a column strip (Fix B).

        Open wells expose the vial interior (dark circle + bright ring) which
        creates a periodic intensity pattern along the column axis.
        Closed lids present a flat, near-uniform surface.

        Quantifies this as the ratio of mid-high frequency power to total power
        in the 1D column-averaged intensity profile.  The result is
        illumination-invariant because it operates in the frequency domain.

        When ``profile_sigma_factor > 0`` a Gaussian kernel is applied across
        the strip width before averaging, de-emphasising edge pixels.  This
        suppresses contamination from the rack's internal cell-wall dividers
        that sometimes fall inside the strip of the leftmost (D1) column,
        because the grid-tracker's line position is at or near the rack border.

        Returns:
            float in [0, 1]: high → textured/open, low → flat/closed
        """
        if len(strip) < 4:
            return 0.0

        w = strip.shape[1] if strip.ndim == 2 else 1

        # Collapse strip width-wise → 1D vertical intensity profile.
        # Optional Gaussian weighting reduces rack-wall edge contamination.
        if self.profile_sigma_factor > 0.0 and w >= 2:
            sigma = max(1.0, w * self.profile_sigma_factor)
            x = np.linspace(-(w - 1) / 2, (w - 1) / 2, w, dtype=np.float32)
            weights = np.exp(-x ** 2 / (2.0 * sigma ** 2))
            weights /= weights.sum()
            profile = (strip.astype(np.float32) @ weights)
        else:
            profile = strip.mean(axis=1).astype(np.float32)

        # Remove DC component so power ratios are not dominated by mean brightness
        profile -= profile.mean()

        spectrum = np.abs(np.fft.rfft(profile)) ** 2
        n = len(spectrum)

        # Mid-high band: skip DC (index 0) and the lowest 1/8 of frequencies
        # to avoid pedestal artefacts, capture up to 75 % of Nyquist.
        low_cut = max(1, n // 8)
        high_cut = max(low_cut + 1, int(n * 0.75))

        mid_high_power = spectrum[low_cut:high_cut].mean() if high_cut > low_cut else 0.0
        total_power = spectrum[1:].mean() + 1e-8  # skip DC, avoid div-by-zero

        return float(mid_high_power / total_power)

    def _pairwise_cluster_classify(
        self,
        column_features: Dict[str, float],
        column_vial_counts: Dict[str, int],
    ) -> Optional[Dict[str, str]]:
        """
        Classify all columns via 1-D gap-based state splitting (Fix C, revised).

        Root cause of the K-Means failure
        -----------------------------------
        K-Means with k=2 *always* forces a binary split regardless of whether a
        real state boundary exists.  When all N columns are in the same state
        (all OPEN or all CLOSED) the natural inter-column FFT variation (caused
        by slight illumination differences, vial density, perspective) can exceed
        the old ``min_feature_separation`` range check, causing K-Means to
        produce a spurious 1:N-1 or 2:N-2 split — systematically wrong.

        Gap-based approach
        -------------------
        Sort all column FFT features.  Find the **largest gap** between adjacent
        sorted values.  Only if that gap exceeds ``min_feature_separation`` is
        there a meaningful state boundary between the two groups.

        - Same state (all OPEN or all CLOSED): inter-column gaps are all small
          (< 0.08).  Method returns None → caller falls back to reference-based.
        - Genuinely mixed (some OPEN, some CLOSED): a large gap separates the two
          groups (typically 0.10–0.25).  Columns below the gap → CLOSED_LID;
          columns above the gap → OPEN_LID (higher texture = visible vial interior).

        Returns None when no significant gap is found.
        """
        col_ids = list(column_features.keys())
        if len(col_ids) < 2:
            return None

        features = np.array([column_features[c] for c in col_ids], dtype=np.float32)

        # Sort columns by ascending FFT feature value
        sorted_idx = np.argsort(features)
        sorted_features = features[sorted_idx]
        sorted_col_ids = [col_ids[int(i)] for i in sorted_idx]

        # Gaps between consecutive sorted feature values
        gaps = np.diff(sorted_features)  # length N-1
        max_gap = float(gaps.max()) if len(gaps) > 0 else 0.0
        split_pos = int(np.argmax(gaps))  # index after which the OPEN group begins

        if max_gap < self.min_feature_separation:
            logger.info(
                f"FFT max-gap {max_gap:.4f} < min_feature_separation "
                f"{self.min_feature_separation:.4f}: no meaningful state boundary found, "
                "all columns appear same-state — falling back to reference-based classification"
            )
            return None

        # Columns at rank ≤ split_pos → CLOSED (lower texture)
        # Columns at rank > split_pos → OPEN  (higher texture)
        column_states: Dict[str, str] = {}
        for rank, col_id in enumerate(sorted_col_ids):
            column_states[col_id] = "OPEN_LID" if rank > split_pos else "CLOSED_LID"

        closed_cols = [c for c, s in column_states.items() if s == "CLOSED_LID"]
        open_cols = [c for c, s in column_states.items() if s == "OPEN_LID"]
        logger.info(
            f"Gap-based FFT split: max_gap={max_gap:.4f} at rank {split_pos}/{len(col_ids)-1} "
            f"(boundary {sorted_features[split_pos]:.4f}→{sorted_features[split_pos+1]:.4f}) | "
            f"CLOSED={closed_cols}, OPEN={open_cols}"
        )
        return column_states

    def classify(
        self,
        frame: np.ndarray,
        dominant_lines: List[Tuple],
        centroids: List[Tuple[int, int]],
    ) -> Dict[str, str]:
        """
        Classify lid state for each column using reference-based subtraction.

        Args:
            frame: Input frame (BGR)
            dominant_lines: List of ((angle_deg, d), count, line_id)
            centroids: List of (x, y) vial centroids

        Returns:
            Dictionary: {column_id: "OPEN_LID" or "CLOSED_LID"}
        """
        if self.debug:
            self.debug_info = {}

        # Step 1: Assign vials to columns
        column_vials = self.assign_vials_to_columns(dominant_lines, centroids)

        if not column_vials:
            logger.warning("No vials assigned to columns")
            return {}

        # Step 2: Extract column strips
        column_strips: Dict[str, np.ndarray] = {}
        line_params: Dict[str, Tuple[float, float]] = {}

        for line_data in dominant_lines:
            if len(line_data) >= 3:
                (angle_deg, d), count, line_id = line_data
            elif len(line_data) >= 2:
                (angle_deg, d), count = line_data[:2]
                line_id = f"D{len(line_params) + 1}"
            else:
                continue

            if line_id not in column_vials:
                continue

            line_params[line_id] = (angle_deg, d)
            vials = column_vials[line_id]

            strip = self.extract_column_strip(frame, vials, angle_deg, d)
            if strip is not None:
                column_strips[line_id] = strip

        if not column_strips:
            logger.warning("No column strips extracted")
            return {}

        # Step 3: Identify D1 as reference (OPEN)
        sorted_columns = sorted(
            column_strips.keys(),
            key=lambda c: int(c[1:]) if len(c) > 1 and c[1:].isdigit() else 999
        )

        if not sorted_columns:
            return {}

        reference_id = sorted_columns[0]  # D1
        reference_strip = column_strips[reference_id]

        logger.info(f"Using {reference_id} as reference ({self.reference_state})")

        # Step 4: Compute differences and vial counts for all columns
        column_differences: Dict[str, float] = {}
        column_vial_counts: Dict[str, int] = {}

        # Count vials per column
        for column_id in column_strips.keys():
            column_vial_counts[column_id] = len(column_vials.get(column_id, []))

        reference_vial_count = column_vial_counts.get(reference_id, 1)

        # Compute image differences
        for column_id, strip in column_strips.items():
            if column_id == reference_id:
                # Reference has zero difference
                column_differences[column_id] = 0.0
            else:
                img_diff = self.compute_difference(reference_strip, strip)
                column_differences[column_id] = img_diff

            if self.debug:
                self.debug_info[column_id] = {
                    'strip': strip,
                    'difference': column_differences[column_id],
                    'vial_count': column_vial_counts[column_id],
                    'is_reference': (column_id == reference_id)
                }

        # Step 5: Classification
        # Primary path (Fix B + C): FFT texture features + K-Means clustering.
        # Falls back to the two-gate reference-subtraction system when all
        # columns appear visually similar (feature range is too small for
        # K-Means to find a meaningful boundary).
        column_states: Dict[str, str] = {}
        self.gateway_info = []

        if self.use_fft_features and self.use_pairwise_clustering:
            # Compute FFT texture feature for every column strip
            column_fft_features: Dict[str, float] = {
                col_id: self._compute_fft_texture_feature(strip)
                for col_id, strip in column_strips.items()
            }

            if self.debug:
                for col_id, fft_val in column_fft_features.items():
                    if col_id in self.debug_info:
                        self.debug_info[col_id]['fft_feature'] = fft_val

            clustering_result = self._pairwise_cluster_classify(
                column_fft_features, column_vial_counts
            )

            if clustering_result is not None:
                # Clustering succeeded — apply Gate 1 structural override then return
                column_states = dict(clustering_result)

                for col_id in sorted(
                    column_strips.keys(),
                    key=lambda c: int(c[1:]) if len(c) > 1 and c[1:].isdigit() else 999,
                ):
                    vial_count = column_vial_counts.get(col_id, 0)
                    vial_ratio = vial_count / max(reference_vial_count, 1)
                    fft_val = column_fft_features.get(col_id, 0.0)
                    raw_state = column_states.get(col_id, self.reference_state)

                    # Gate 1 structural override: very few vials → force OPEN
                    gate_1_passed = vial_ratio <= self.vial_count_threshold
                    if gate_1_passed:
                        column_states[col_id] = "OPEN_LID"

                    final_state = column_states.get(col_id, raw_state)
                    self.gateway_info.append({
                        'column_id': col_id,
                        'classification_method': 'gap_fft',
                        'fft_feature': float(fft_val),
                        'vial_count': vial_count,
                        'reference_vial_count': reference_vial_count,
                        'gate_1_evaluated': True,
                        'gate_1_passed': gate_1_passed,
                        'gate_1_vial_ratio': float(vial_ratio),
                        'vial_count_threshold': float(self.vial_count_threshold),
                        'raw_state': raw_state,
                        'final_state': final_state,
                        'reason': (
                            f"Gate 1 override: vial_ratio={vial_ratio:.2f}" if gate_1_passed
                            else f"K-Means FFT: fft={fft_val:.4f}"
                        ),
                    })

                    if self.debug and col_id in self.debug_info:
                        self.debug_info[col_id]['state'] = final_state
                        self.debug_info[col_id]['gate'] = (
                            'Gate 1 (Structural)' if gate_1_passed else 'K-Means FFT'
                        )

                logger.info(f"Classified {len(column_states)} columns via K-Means FFT: {column_states}")
                stabilized = self.update_temporal_buffer(column_states)
                for entry in self.gateway_info:
                    cid = entry['column_id']
                    if cid in stabilized:
                        entry['stabilized_state'] = stabilized[cid]
                        if cid in self.column_buffers:
                            entry['temporal_buffer'] = list(self.column_buffers[cid])
                            entry['temporal_buffer_size'] = len(self.column_buffers[cid])
                return stabilized
            # else: clustering returned None → fall through to two-gate system

        # --- Fallback: Two-Gate Reference-Subtraction System ---
        # D1 is always the reference with configured state
        column_states[reference_id] = self.reference_state

        if len(column_differences) == 1:
            # Only one column detected
            return column_states

        # Get differences for non-reference columns
        other_diffs = [diff for col_id, diff in column_differences.items() if col_id != reference_id]

        if not other_diffs:
            return column_states

        # === Smart Threshold Determination ===
        diff_array = np.array(other_diffs, dtype=np.float32)
        
        if self.diff_threshold is not None and self.diff_threshold_method == "fixed":
            # Use fixed threshold provided by user
            visual_threshold = self.diff_threshold
            logger.info(f"Using fixed visual threshold: {visual_threshold:.2f}")
        
        elif self.diff_threshold_method == "percentile":
            # Use 60th percentile (robust, splits distributions well)
            visual_threshold = np.percentile(diff_array, 60)
            logger.info(f"Using percentile-based threshold (60th): {visual_threshold:.2f}")
        
        elif self.diff_threshold_method == "median_iqr":
            # Median + iqr_multiplier * IQR (very robust to outliers)
            q1 = np.percentile(diff_array, 25)
            q3 = np.percentile(diff_array, 75)
            iqr = q3 - q1
            median = np.median(diff_array)
            visual_threshold = median + self.iqr_multiplier * iqr
            logger.info(f"Using median+IQR threshold: {visual_threshold:.2f} (median={median:.2f}, IQR={iqr:.2f}, multiplier={self.iqr_multiplier})")
        
        else:
            # Fallback: median as default
            visual_threshold = np.median(diff_array)
            logger.info(f"Using median threshold (fallback): {visual_threshold:.2f}")

        # === Minimum Threshold Floor ===
        # When N is small (N=2), median+IQR collapses to max(diffs), causing the
        # highest-diff column to always be called OPEN. The floor prevents this.
        if self.min_visual_threshold > 0 and visual_threshold < self.min_visual_threshold:
            logger.info(f"Raising threshold from {visual_threshold:.2f} to floor {self.min_visual_threshold:.2f} (N={len(other_diffs)})")
            visual_threshold = self.min_visual_threshold

        # === Two-Gate Classification ===
        # Clear gateway info for this classification
        self.gateway_info = []
        
        for column_id in sorted(column_strips.keys(), 
                               key=lambda c: int(c[1:]) if len(c) > 1 and c[1:].isdigit() else 999):
            if column_id == reference_id:
                # Reference column - always uses reference state
                gateway_entry = {
                    'column_id': column_id,
                    'is_reference': True,
                    'reference_state': self.reference_state,
                    'vial_count': column_vial_counts[column_id],
                    'reference_vial_count': reference_vial_count,
                    'mean_difference': 0.0,
                    'visual_threshold': visual_threshold,
                    'vial_count_threshold': self.vial_count_threshold,
                    'gate_1_evaluated': False,
                    'gate_1_passed': None,
                    'gate_1_vial_ratio': None,
                    'gate_2_evaluated': False,
                    'gate_2_passed': None,
                    'gate_2_mean_diff': None,
                    'gate_used': 'Reference (no gate evaluation)',
                    'final_state': self.reference_state,
                    'reason': f'Reference column ({self.reference_state})'
                }
                self.gateway_info.append(gateway_entry)
                column_states[column_id] = self.reference_state
                continue

            vial_count = column_vial_counts[column_id]
            vial_ratio = vial_count / max(reference_vial_count, 1)
            mean_diff = column_differences[column_id]

            # Initialize gateway entry for this column
            gateway_entry = {
                'column_id': column_id,
                'is_reference': False,
                'reference_state': self.reference_state,
                'vial_count': vial_count,
                'reference_vial_count': reference_vial_count,
                'mean_difference': float(mean_diff),
                'visual_threshold': float(visual_threshold),
                'vial_count_threshold': float(self.vial_count_threshold),
                'gate_1_evaluated': True,
                'gate_1_passed': None,
                'gate_1_vial_ratio': float(vial_ratio),
                'gate_2_evaluated': False,
                'gate_2_passed': None,
                'gate_2_mean_diff': None,
                'gate_used': None,
                'final_state': None,
                'reason': None
            }

            # --- GATE 1: Structural Check (Hard Override) ---
            # Simple relative ratio check
            if vial_ratio <= self.vial_count_threshold:
                # Very few vials detected → OPEN_LID
                state = "OPEN_LID"
                gate = "Gate 1 (Structural)"
                reason = (f"Low vial ratio: {vial_count}/{reference_vial_count} = {vial_ratio:.2f} "
                         f"≤ {self.vial_count_threshold} threshold")
                
                gateway_entry['gate_1_passed'] = True
                gateway_entry['gate_used'] = gate
                gateway_entry['final_state'] = state
                gateway_entry['reason'] = reason
            
            else:
                # Gate 1 did not pass - proceed to Gate 2
                gateway_entry['gate_1_passed'] = False
                gateway_entry['gate_2_evaluated'] = True
                gateway_entry['gate_2_mean_diff'] = float(mean_diff)
                
                # --- GATE 2: Visual Similarity Check ---
                if mean_diff < visual_threshold:
                    # Similar to reference → Same state as reference
                    state = self.reference_state
                    gate = "Gate 2 (Visual)"
                    reason = f"Similar to ref: diff={mean_diff:.2f} < thresh={visual_threshold:.2f}"
                    
                    gateway_entry['gate_2_passed'] = True
                    gateway_entry['gate_used'] = gate
                    gateway_entry['final_state'] = state
                    gateway_entry['reason'] = reason
                else:
                    # Different from reference → Opposite state
                    state = "OPEN_LID" if self.reference_state == "CLOSED_LID" else "CLOSED_LID"
                    gate = "Gate 2 (Visual)"
                    reason = f"Different from ref: diff={mean_diff:.2f} ≥ thresh={visual_threshold:.2f}"
                    
                    gateway_entry['gate_2_passed'] = False
                    gateway_entry['gate_used'] = gate
                    gateway_entry['final_state'] = state
                    gateway_entry['reason'] = reason

            column_states[column_id] = state
            self.gateway_info.append(gateway_entry)

            # Store debug info
            if self.debug and column_id in self.debug_info:
                self.debug_info[column_id]['visual_threshold'] = visual_threshold
                self.debug_info[column_id]['vial_threshold'] = self.vial_count_threshold
                self.debug_info[column_id]['state'] = state
                self.debug_info[column_id]['gate'] = gate
                self.debug_info[column_id]['reason'] = reason
                self.debug_info[column_id]['vial_ratio'] = vial_ratio

        logger.info(f"Classified {len(column_states)} columns: {column_states}")

        # Step 6: Temporal smoothing
        stabilized_states = self.update_temporal_buffer(column_states)
        
        # Update gateway_info with stabilized states (what's actually displayed)
        for gateway_entry in self.gateway_info:
            column_id = gateway_entry['column_id']
            if column_id in stabilized_states:
                # Store raw state (before temporal smoothing)
                gateway_entry['raw_state'] = gateway_entry['final_state']
                # Store stabilized state (after temporal smoothing - what's displayed)
                gateway_entry['stabilized_state'] = stabilized_states[column_id]
                # Store buffer info for debugging
                if column_id in self.column_buffers:
                    buffer = list(self.column_buffers[column_id])
                    gateway_entry['temporal_buffer'] = buffer
                    gateway_entry['temporal_buffer_size'] = len(buffer)
                # Update final_state to match stabilized (what's displayed in image)
                gateway_entry['final_state'] = stabilized_states[column_id]

        return stabilized_states

    def update_temporal_buffer(
        self, current_states: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Apply temporal smoothing using majority voting over buffer_size frames.

        Args:
            current_states: Current frame's classification results

        Returns:
            Stabilized states after temporal smoothing
        """
        stabilized_states: Dict[str, str] = {}

        for column_id, state in current_states.items():
            if column_id not in self.column_buffers:
                self.column_buffers[column_id] = deque(maxlen=self.buffer_size)

            self.column_buffers[column_id].append(state)

            # Majority vote
            buffer = list(self.column_buffers[column_id])
            open_count = buffer.count("OPEN_LID")
            closed_count = buffer.count("CLOSED_LID")

            stabilized_states[column_id] = (
                "OPEN_LID" if open_count > closed_count else "CLOSED_LID"
            )

        return stabilized_states

    def save_classification_overview_figure(
        self, output_path: str, column_states: Dict[str, str]
    ) -> bool:
        """
        Save debug figure showing column strips, differences, and classifications.

        Args:
            output_path: Path to save the figure
            column_states: Final classification results

        Returns:
            True if successful, False otherwise
        """
        if not self.debug or not self.debug_info:
            return False

        try:
            sorted_cols = sorted(
                self.debug_info.keys(),
                key=lambda c: int(c[1:]) if len(c) > 1 and c[1:].isdigit() else 999
            )

            if not sorted_cols:
                return False

            n_cols = len(sorted_cols)
            fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

            if n_cols == 1:
                axes = axes.reshape(2, 1)

            for idx, col_id in enumerate(sorted_cols):
                col_debug = self.debug_info[col_id]
                strip = col_debug.get('strip')
                diff = col_debug.get('difference', 0.0)
                is_ref = col_debug.get('is_reference', False)
                state = column_states.get(col_id, "UNKNOWN")

                # Top row: Column strip
                ax_strip = axes[0, idx]
                if strip is not None:
                    ax_strip.imshow(strip, cmap='gray')
                    ax_strip.axis('off')
                    
                    title_text = f"{col_id} - {'REF' if is_ref else state}"
                    title_color = 'blue' if is_ref else ('green' if state == 'OPEN_LID' else 'red')
                    ax_strip.set_title(title_text, color=title_color, fontsize=10, weight='bold')
                else:
                    ax_strip.text(0.5, 0.5, f"{col_id}\nNo data", ha='center', va='center')
                    ax_strip.axis('off')

                # Bottom row: Two-gate classification info
                ax_diff = axes[1, idx]
                ax_diff.axis('off')
                
                vial_count = col_debug.get('vial_count', 0)
                
                if is_ref:
                    ref_label = "OPEN" if self.reference_state == "OPEN_LID" else "CLOSED"
                    ax_diff.text(0.5, 0.5, 
                                f"REFERENCE\n({ref_label})\ndiff = 0.00\nVials: {vial_count}",
                                ha='center', va='center', fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                else:
                    gate = col_debug.get('gate', 'Unknown')
                    reason = col_debug.get('reason', 'No reason')
                    vial_ratio = col_debug.get('vial_ratio', 1.0)
                    visual_thresh = col_debug.get('visual_threshold', 0.0)
                    vial_thresh = col_debug.get('vial_threshold', 0.5)
                    
                    # Format text based on which gate was used
                    if 'Gate 1' in gate:
                        # Structural gate - show relative ratio only
                        text = (f"🚪 {gate}\n"
                               f"Vials: {vial_count}\n"
                               f"Ratio: {vial_ratio:.2f} (≤{vial_thresh}?)\n"
                               f"→ {state.replace('_LID', '')}")
                    else:
                        # Visual gate
                        text = (f"👁️ {gate}\n"
                               f"Diff: {diff:.2f} | Thresh: {visual_thresh:.2f}\n"
                               f"Vials: {vial_count} (ratio={vial_ratio:.2f})\n"
                               f"→ {state.replace('_LID', '')}")
                    
                    ax_diff.text(0.5, 0.5, text,
                                ha='center', va='center', fontsize=7,
                                bbox=dict(boxstyle='round', 
                                         facecolor='lightgreen' if state == 'OPEN_LID' else 'lightcoral',
                                         alpha=0.7))

            ref_label = "OPEN" if self.reference_state == "OPEN_LID" else "CLOSED"
            fig.suptitle(f"Reference-Based Lid Classification (D1 = {ref_label} Reference)", 
                        fontsize=12, weight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved classification overview: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving overview figure: {e}", exc_info=True)
            return False

    def reset_buffers(self):
        """Reset all temporal buffers."""
        self.column_buffers.clear()
        logger.info("Temporal buffers reset")
    
    def get_gateway_info(self) -> List[Dict]:
        """
        Get gateway interaction information for the last classification.
        
        Returns:
            List of dictionaries, each containing gateway interaction details for one column
        """
        return self.gateway_info.copy()