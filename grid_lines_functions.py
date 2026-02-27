import numpy as np


def compute_half_angle_histogram_for_point1(points, index, bin_width=6, overlap=0.5):
    p = points[index]
    n = len(points)
    # Calculate bin parameters
    step_size = bin_width * (1 - overlap)  # Distance between bin starts
    nbins = int(180 / step_size)
    # Create overlapping bins - using numpy array instead of list
    bin_edges = np.array([i * step_size for i in range(nbins + 1)])
    hist = np.zeros(nbins)
    angles = []
    distances = []
    # Collect angles and distances
    for j in range(n):
        if j == index:
            continue
        q = points[j]
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        distance = np.sqrt(dx**2 + dy**2)
        theta = np.degrees(np.arctan2(dy, dx))
        # Normalize angle to [0,360)
        if theta < 0:
            theta += 360
        # Fold angle into [0,180)
        theta = theta % 180
        angles.append(theta)
        distances.append(distance)
        # Add point to all overlapping bins
        for i in range(nbins):
            bin_start = bin_edges[i]
            bin_end = bin_start + bin_width
            if bin_start <= theta <= bin_end:
                hist[i] += 1
                # # print(f"Angle {theta:.1f}° added to bin {i} ({bin_start:.1f}°-{bin_end:.1f}°)")
    return hist, bin_edges, angles, distances


def get_local_dominant_angles1_optimized(points, bin_width=6, distance_threshold=200.0, debug=False, frame_count=0):
    def is_horizontal(angle):
        angle = angle % 180
        return angle <= 45 or angle >= 135

    def is_vertical(angle):
        angle = angle % 180
        return 45 < angle < 135

    def is_valid_angle(angle):
        return is_horizontal(angle) or is_vertical(angle)

    def normalize_horizontal_angles(angles):
        normalized = []
        for angle in angles:
            if angle <= 45:  # If angle is near 0°
                normalized.append(angle + 180)  # Convert to equivalent angle near 180°
            else:
                normalized.append(angle)
        return normalized

    def remove_outliers(angles, threshold=10.0):
        if not angles:
            return angles
        # For horizontal angles, normalize them first
        normalized = normalize_horizontal_angles(angles)
        median = np.median(normalized)
        # Keep angles that are within threshold of median
        filtered = []
        for orig, norm in zip(angles, normalized):
            if abs(norm - median) <= threshold:
                filtered.append(orig)
        return filtered if filtered else angles  # Return original if all removed

    # Sample multiple points
    sample_size = min(9, len(points))
    sampled_angles = []
    for i in range(sample_size):
        hist, bin_edges, angles, distances = compute_half_angle_histogram_for_point1(
            points, i, bin_width
        )
        # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        peak_idx = np.argmax(hist)
        dominant_bin_start = bin_edges[peak_idx]
        dominant_bin_end = bin_edges[peak_idx + 1]
        closest_points = [
            (angle, dist)
            for angle, dist in zip(angles, distances)
            if dominant_bin_start <= angle < dominant_bin_end and dist <= distance_threshold
        ]
        if closest_points:
            bin_angles, bin_distances = zip(*closest_points)
            weights = [1 / d if d > 0 else 1 for d in bin_distances]
            angle = np.average(bin_angles, weights=weights)
            sampled_angles.append(angle)
    if not sampled_angles:
        final_angle = 180
    else:
        # Group angles by their type
        horizontal_angles = [a for a in sampled_angles if is_horizontal(a)]
        vertical_angles = [a for a in sampled_angles if is_vertical(a)]
        if len(horizontal_angles) >= len(vertical_angles):
            # Remove outliers from horizontal angles
            filtered_angles = remove_outliers(horizontal_angles)
            # Normalize and average the filtered angles
            normalized_angles = normalize_horizontal_angles(filtered_angles)
            final_angle = np.mean(normalized_angles) % 180
        else:
            # Remove outliers from vertical angles
            filtered_angles = remove_outliers(vertical_angles)
            final_angle = np.mean(filtered_angles)
    # Assign final angle to all points
    filtered_dominant_angles = [
        (points[i][0], points[i][1], final_angle)
        for i in range(len(points))
    ]
    return filtered_dominant_angles


def generate_recessive_angles(dominant_angles):
    recessive_lines = []
    for x, y, dominant_angle in dominant_angles:
        recessive_angle = (dominant_angle + 90) % 180
        rec_line_angle, rec_line_d = angle_to_line_params(x, y, recessive_angle)
        recessive_lines.append(((rec_line_angle, rec_line_d), recessive_angle))
    return recessive_lines


def angle_to_line_params(x, y, angle_deg):
    # Convert direction angle to a line in normal form
    # normal angle (theta_n) = angle_deg + 90°
    theta_n = np.radians(angle_deg + 90)
    d = x * np.cos(theta_n) + y * np.sin(theta_n)
    return angle_deg, d


def merge_lines_pick_max_support(lines, angle_threshold=20.0, distance_threshold=30.0):
    if not lines:
        return []
    merged = []
    used = set()
    for i in range(len(lines)):
        if i in used:
            continue
        current_line, current_dist = lines[i]
        current_angle, current_d = current_line
        # Find similar lines
        similar_indices = []
        for j in range(len(lines)):
            if j in used:
                continue
            other_line, other_dist = lines[j]
            other_angle, other_d = other_line
            # Check if lines are similar
            angle_diff = abs(current_angle - other_angle) % 180
            angle_diff = min(angle_diff, 180 - angle_diff)
            dist_diff = abs(current_d - other_d)
            if angle_diff < angle_threshold and dist_diff < distance_threshold:
                similar_indices.append(j)
                used.add(j)
        # Instead of averaging parameters, pick the line with most support
        best_line = max([(lines[idx], idx) for idx in similar_indices],
                        key=lambda x: x[0][1])  # use distance as support metric
        merged.append(best_line[0])
        used.add(i)
    return merged


def filter_top_percent(merged_lines, percent=5):
    if not merged_lines:
        # # print("No merged lines to filter.")
        return []
    if percent <= 0:
        # # print("Percent must be greater than 0. Returning the original merged_lines.")
        return merged_lines
    if percent >= 100:
        # # print("Percent must be less than 100. Returning an empty list.")
        return []
    # Extract all points_dist values
    points_dist_values = [line[1] for line in merged_lines]
    # Calculate the cutoff threshold using the percentile
    cutoff = np.percentile(points_dist_values, 100 - percent)
    # Filter out lines with points_dist greater than the cutoff
    filtered_lines = [line for line in merged_lines if line[1] <= cutoff]
    return filtered_lines


def compute_best_angle_for_anchor(points, anchor_index, bin_width=6, distance_threshold=100.0, debug=False):
    # -- Same internal logic as in your 'compute_half_angle_histogram_for_point1'
    # anchor = points[anchor_index]
    # Build histogram
    hist, bin_edges, angles, distances = compute_half_angle_histogram_for_point1(
        points, anchor_index, bin_width
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Find the dominant bin
    peak_idx = np.argmax(hist)
    dominant_bin_start = bin_edges[peak_idx]
    dominant_bin_end = bin_edges[peak_idx + 1]

    # Select points (relative to the first point) within the dominant bin and distance threshold
    closest_points = [
        (angle, dist)
        for angle, dist in zip(angles, distances)
        if dominant_bin_start <= angle < dominant_bin_end and dist <= distance_threshold
    ]
    if closest_points:
        # Extract angles and distances
        bin_angles, bin_distances = zip(*closest_points)
        # Compute weighted average angle (weighted by 1/distance)
        weights = [1 / d if d > 0 else 1 for d in bin_distances]  # Avoid division by zero
        best_angle = np.average(bin_angles, weights=weights)
    else:
        best_angle = bin_centers[peak_idx]
    return best_angle


def get_local_dominant_angle_multi_anchors(points,
                                           num_anchors=5,
                                           bin_width=6,
                                           distance_threshold=100.0,
                                           outlier_threshold=5.0,
                                           debug=False):
    from random import sample
    n = len(points)
    if n == 0:
        # No points at all, no angle
        return 0.0
    # If too few points, just use them all:
    if n < num_anchors:
        anchor_indices = range(n)
    else:
        anchor_indices = sample(range(n), num_anchors)
    # Collect angles from each anchor
    anchor_angles = []
    for anchor_idx in anchor_indices:
        angle = compute_best_angle_for_anchor(
            points, anchor_idx, bin_width=bin_width,
            distance_threshold=distance_threshold, debug=debug
        )
        anchor_angles.append(angle % 180.0)  # keep in [0,180)
    # --- Step 1: compute the median angle ---
    median_angle = np.median(anchor_angles)
    # --- Step 2: compute absolute differences from median ---
    abs_diffs = [abs((a - median_angle + 90) % 180 - 90)
                 for a in anchor_angles]
    # --- Step 3: filter out outliers ---
    filtered_angles = []
    for angle, diff in zip(anchor_angles, abs_diffs):
        if diff <= outlier_threshold:
            filtered_angles.append(angle)
    # if all angles got thrown out (worst case), just keep them all
    if len(filtered_angles) == 0:
        filtered_angles = anchor_angles
    # --- Step 4: average the remaining angles ---
    final_angle = np.mean(filtered_angles) % 180.0
    return final_angle


def get_local_dominant_angles_multi_anchors_optimized(points,
                                                      num_anchors=5,
                                                      bin_width=6,
                                                      distance_threshold=100.0,
                                                      angle_tolerance=10.0,
                                                      debug=False):
    # Step 1: get a single best angle from multiple anchors
    best_angle = get_local_dominant_angle_multi_anchors(
        points,
        num_anchors=num_anchors,
        bin_width=bin_width,
        distance_threshold=distance_threshold,
        debug=debug
    )
    # Step 2: assign that angle to every point

    # median_direction = best_angle
    # Step 3: (optional) filter out angles if needed
    all_dominant_angles = []
    for (x, y) in points:
        all_dominant_angles.append((x, y, best_angle))
    return all_dominant_angles


def get_local_dominant_angle_multi_anchors_folded(points,
                                                  num_anchors=5,
                                                  bin_width=6,
                                                  distance_threshold=100.0,
                                                  outlier_threshold=5.0,
                                                  debug=False):
    from random import sample

    def fold_angle_0_90(angle_deg):
        return min(angle_deg, 180 - angle_deg)
    n = len(points)
    if n == 0:
        return 0.0
    if n < num_anchors:
        anchor_indices = range(n)
    else:
        anchor_indices = sample(range(n), num_anchors)
    # 1. compute anchor angles
    anchor_angles = []
    for idx in anchor_indices:
        angle = compute_best_angle_for_anchor(
            points, idx, bin_width=bin_width, distance_threshold=distance_threshold, debug=False
        )
        # store angle mod 180:
        anchor_angles.append(angle % 180.0)
    # 2. fold to [0,90]
    folded_angles = [fold_angle_0_90(a) for a in anchor_angles]
    # 3. remove outliers around median
    median_val = np.median(folded_angles)

    def circular_diff(a, b):
        return abs(a - b)
    filtered = []
    for a in folded_angles:
        if circular_diff(a, median_val) <= outlier_threshold:
            filtered.append(a)
    if len(filtered) == 0:
        filtered = folded_angles
    final_angle_folded = np.mean(filtered)
    all_dominant_angles = []
    for (x, y) in points:
        all_dominant_angles.append((x, y, final_angle_folded))

    return all_dominant_angles
