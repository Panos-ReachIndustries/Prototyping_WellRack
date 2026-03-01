"""
Generate pseudo vial-keypoint annotations from the existing row-line
annotations in lid_annotations.json.

Strategy:
    1. Each row annotation gives a horizontal line through vial centres.
    2. Estimate column positions by dividing the line length into N equal
       segments (N=12 for a standard 96-well rack, N=8 for 8-column racks).
    3. Write per-image keypoint lists to vial_keypoints.json.

This produces approximate labels for bootstrapping training; they should
be refined using the annotation tool (annotate_tool.py) for best results.

Usage:
    python -m keypoint_model.generate_pseudo_labels \
        --lid-annotations assets/testing_images/lid_annotations.json \
        --output assets/testing_images/vial_keypoints.json \
        --num-cols 12
"""
import argparse
import json
import os

import cv2
import numpy as np


def generate_keypoints_from_lines(lines: list[dict],
                                  num_cols: int = 12,
                                  img_shape: tuple | None = None
                                  ) -> list[list[float]]:
    """
    Given row-line annotations, interpolate column positions along each line.
    """
    keypoints = []
    for entry in lines:
        (x1, y1), (x2, y2) = entry["line"]
        for j in range(num_cols):
            t = (j + 0.5) / num_cols
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            keypoints.append([round(x, 1), round(y, 1)])
    return keypoints


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo vial keypoints from row-line annotations"
    )
    parser.add_argument("--lid-annotations", type=str,
                        default="assets/testing_images/lid_annotations.json")
    parser.add_argument("--images-dir", type=str,
                        default="assets/testing_images")
    parser.add_argument("--output", type=str,
                        default="assets/testing_images/vial_keypoints.json")
    parser.add_argument("--num-cols", type=int, default=12,
                        help="Number of vial columns per row (default 12)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--vis-dir", type=str, default="pseudo_label_vis")
    args = parser.parse_args()

    with open(args.lid_annotations) as f:
        lid_data = json.load(f)

    vial_keypoints = {}
    for fname, lines in lid_data.items():
        kps = generate_keypoints_from_lines(lines, num_cols=args.num_cols)
        vial_keypoints[fname] = kps
        print(f"{fname}: {len(lines)} rows → {len(kps)} keypoints")

    with open(args.output, "w") as f:
        json.dump(vial_keypoints, f, indent=2)
    print(f"\nSaved {len(vial_keypoints)} entries → {args.output}")

    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)
        for fname, kps in vial_keypoints.items():
            img_path = os.path.join(args.images_dir, fname)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            for (x, y) in kps:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            out_path = os.path.join(args.vis_dir, fname)
            cv2.imwrite(out_path, img)
        print(f"Visualizations saved → {args.vis_dir}/")


if __name__ == "__main__":
    main()
