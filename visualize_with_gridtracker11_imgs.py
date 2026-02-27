"""
visualize_with_gridtracker11_imgs.py
====================================
Image batch visualization with GridTracker v11 and evaluation metrics.

Runs the same logic as visualize_with_gridtracker11_video.py but processes images
from an asset folder instead of video. Provides evaluation metrics and results
when GT annotations (lid_annotations.json) are available.

Uses v11 improvements:
- Fix B: FFT texture features
- Fix C: Pairwise K-Means clustering
- Fix D: Per-frame lid classification
- Fix E: Gaussian profile weighting
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from visualize_with_gridtracker11_video import (
    GridTrackerWithExternalMask,
    convert_windows_path_to_wsl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def _lazy_import_evaluation_functions():
    """
    Lazy import of evaluation functions to avoid circular import.
    Returns a dict with the functions, or None if import fails.
    """
    try:
        import importlib.util

        eval_path = Path(__file__).parent / "evaluate_lid_detection.py"
        if eval_path.exists():
            spec = importlib.util.spec_from_file_location(
                "evaluate_lid_detection", eval_path
            )
            eval_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_module)
            return {
                "load_gt_annotations": eval_module.load_gt_annotations,
                "match_lines_to_gt": eval_module.match_lines_to_gt,
                "line_params_to_segment": eval_module.line_params_to_segment,
                "evaluate_images": eval_module.evaluate_images,
                "generate_plots": eval_module.generate_plots,
            }
    except Exception as e:
        logger.warning(
            f"Could not import evaluation functions: {e}. "
            "Evaluation features will be disabled."
        )
        return None
    return None


def process_images_from_folder(
    input_folder: str,
    output_folder: Optional[str] = None,
    show: bool = True,
    debug_output_folder: Optional[str] = None,
    enable_debug: bool = False,
    gt_annotations_path: Optional[str] = None,
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
    """
    Process images from a folder with GridTracker v11 overlay and evaluation.

    Args:
        input_folder: Path to folder containing images
        output_folder: Optional output folder path (default: input_folder/folder_GridTracker11_results)
        show: Whether to display images during processing
        debug_output_folder: Optional folder path to save debug images automatically
        enable_debug: Enable debug display mode
        gt_annotations_path: Optional path to GT annotations JSON (lid_annotations.json)
        enable_lid_classification: Enable lid state classification
        lid_classifier_*: LidStateClassifier parameters (v11 defaults)
    """
    input_folder = convert_windows_path_to_wsl(input_folder)
    if output_folder:
        output_folder = convert_windows_path_to_wsl(output_folder)
    if gt_annotations_path:
        gt_annotations_path = convert_windows_path_to_wsl(gt_annotations_path)

    input_path = Path(input_folder)
    if not input_path.exists():
        raise RuntimeError(f"Input folder does not exist: {input_folder}")

    if output_folder is None:
        output_path = input_path / "folder_GridTracker11_results"
    else:
        output_path = Path(output_folder)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_path}")

    if not gt_annotations_path:
        possible_names = ["lid_annotations.json", "lid_annotATIONS.JSON"]
        for name in possible_names:
            candidate_path = input_path / name
            if candidate_path.exists():
                gt_annotations_path = str(candidate_path)
                gt_annotations_path = convert_windows_path_to_wsl(gt_annotations_path)
                logger.info(f"Auto-detected GT annotations: {gt_annotations_path}")
                break
        if not gt_annotations_path:
            logger.info(
                f"No GT annotations file found. Tried: {possible_names}. "
                "Use --gt-annotations to enable evaluation."
            )

    gt_annotations = None
    eval_funcs = None
    if gt_annotations_path:
        eval_funcs = _lazy_import_evaluation_functions()
        if eval_funcs and eval_funcs.get("load_gt_annotations"):
            try:
                gt_annotations = eval_funcs["load_gt_annotations"](gt_annotations_path)
                logger.info(f"Loaded GT annotations for {len(gt_annotations)} images")
            except Exception as e:
                logger.warning(
                    f"Failed to load GT annotations: {e}. Continuing without evaluation."
                )
                gt_annotations = None
        else:
            logger.warning(
                "GT annotations path provided but evaluation functions not available."
            )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = [
        f
        for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    image_files.sort()

    if not image_files:
        raise RuntimeError(f"No image files found in: {input_folder}")

    logger.info(f"Found {len(image_files)} images to process")

    all_vials_times = []
    all_grid_times = []
    all_total_times = []
    evaluation_data = [] if gt_annotations else None

    for idx, image_file in enumerate(image_files):
        grid_tracker = GridTrackerWithExternalMask(
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

        grid_tracker.detector.input_folder = str(input_path)

        if enable_lid_classification and grid_tracker.lid_classifier is not None:
            grid_tracker.lid_classifier.buffer_size = 1
            grid_tracker.lid_classifier.reset_buffers()

        if debug_output_folder is not None or enable_debug:
            if debug_output_folder is None:
                debug_output_folder = str(output_path / "debug_images")
            grid_tracker.debug_output_folder = debug_output_folder
            grid_tracker.debug_image_name = image_file.stem
            os.makedirs(debug_output_folder, exist_ok=True)
            logger.info(
                f"Debug images for {image_file.name} saved to: {debug_output_folder}"
            )
        else:
            default_debug_folder = output_path / "debug_images"
            grid_tracker.debug_output_folder = str(default_debug_folder)
            grid_tracker.debug_image_name = image_file.stem
            os.makedirs(default_debug_folder, exist_ok=True)

        logger.info(f"Processing image {idx + 1}/{len(image_files)}: {image_file.name}")

        frame = cv2.imread(str(image_file))
        if frame is None:
            logger.warning(f"Could not load image: {image_file.name}, skipping...")
            continue

        height, width = frame.shape[:2]
        plate_mask = np.ones((height, width), dtype=np.uint8) * 255

        grid_tracker.track_frame_with_mask(frame, plate_mask)

        if hasattr(
            grid_tracker.detector, "last_vials_detection_time"
        ) and hasattr(grid_tracker.detector, "last_grid_generation_time"):
            all_vials_times.append(grid_tracker.detector.last_vials_detection_time)
            all_grid_times.append(grid_tracker.detector.last_grid_generation_time)
            all_total_times.append(
                grid_tracker.detector.last_vials_detection_time
                + grid_tracker.detector.last_grid_generation_time
            )

        gt_lines_for_image = None
        if gt_annotations and image_file.name in gt_annotations:
            gt_lines_for_image = gt_annotations[image_file.name]

        annotated_frame = grid_tracker.draw_grid_overlay(frame)

        frame_height = annotated_frame.shape[0]
        confidence = grid_tracker.get_confidence()
        num_grid_points = len(grid_tracker.current_grid_points)
        num_dominant_lines = len(grid_tracker.dominant_lines)
        lid_states = getattr(grid_tracker, "lid_states", {}) or {}
        num_classified_columns = len(lid_states)
        num_open = sum(1 for s in lid_states.values() if s == "OPEN_LID")
        num_closed = sum(1 for s in lid_states.values() if s == "CLOSED_LID")

        processing_warnings = []
        if num_grid_points == 0:
            processing_warnings.append("NO_GRID_POINTS")
        if num_dominant_lines == 0:
            processing_warnings.append("NO_DOMINANT_LINES")
        if num_classified_columns == 0:
            processing_warnings.append("NO_LID_CLASSIFICATION")
        if confidence < 0.1:
            processing_warnings.append("LOW_CONFIDENCE")

        cv2.putText(
            annotated_frame,
            f"Grid Conf: {confidence:.3f}",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if confidence > 0.3 else (0, 165, 255),
            2,
        )

        if processing_warnings:
            warning_text = " | ".join(processing_warnings)
            cv2.putText(
                annotated_frame,
                f"WARNINGS: {warning_text}",
                (10, frame_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )
            logger.warning(f"Image {image_file.name}: {warning_text}")
        else:
            info_text = (
                f"Points: {num_grid_points} | Lines: {num_dominant_lines} | "
                f"Classified: {num_classified_columns} (O:{num_open} C:{num_closed})"
            )
            cv2.putText(
                annotated_frame,
                info_text,
                (10, frame_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        cv2.putText(
            annotated_frame,
            "Legend: Open (Green) | Closed (Red)",
            (10, frame_height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        output_file = output_path / f"{image_file.stem}_gridtracker11.png"
        success = cv2.imwrite(str(output_file), annotated_frame)
        if not success:
            logger.error(f"Failed to save annotated image: {output_file}")
        else:
            logger.info(f"Saved annotated image to: {output_file}")

        if (
            gt_annotations
            and image_file.name in gt_annotations
            and eval_funcs
            and eval_funcs.get("match_lines_to_gt")
        ):
            try:
                computed_lines = grid_tracker.dominant_lines
                plate_bbox = None
                if grid_tracker.prev_plate_mask is not None:
                    x, y, w, h = cv2.boundingRect(grid_tracker.prev_plate_mask)
                    plate_bbox = (x, y, w, h)

                matches = eval_funcs["match_lines_to_gt"](
                    computed_lines,
                    gt_lines_for_image,
                    (height, width),
                    plate_bbox,
                )

                image_line_distances = []
                image_classification_results = []
                for gt_idx, distance, comp_id, gt_label in matches:
                    if gt_idx is not None:
                        image_line_distances.append(distance)

                    if gt_idx is not None and comp_id in lid_states:
                        computed_label = lid_states[comp_id]
                        if computed_label == "OPEN_LID":
                            computed_label_norm = "open"
                        elif computed_label == "CLOSED_LID":
                            computed_label_norm = "closed"
                        else:
                            computed_label_norm = computed_label.lower()
                        gt_label_norm = gt_label.lower() if gt_label else None
                        is_correct = (
                            computed_label_norm == gt_label_norm if gt_label_norm else False
                        )
                        image_classification_results.append(
                            {
                                "line_id": comp_id,
                                "computed": computed_label_norm,
                                "gt": gt_label_norm,
                                "correct": is_correct,
                            }
                        )

                evaluation_data.append(
                    {
                        "image": image_file.name,
                        "num_gt_lines": len(gt_lines_for_image),
                        "num_computed_lines": len(computed_lines),
                        "num_matched_lines": len([m for m in matches if m[0] is not None]),
                        "line_distances": image_line_distances,
                        "classification_results": image_classification_results,
                        "mean_line_distance": (
                            np.mean(image_line_distances)
                            if image_line_distances
                            else float("nan")
                        ),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate image {image_file.name}: {e}", exc_info=True
                )

        if show:
            cv2.imshow("WellRack GridTracker v11 Overlay", annotated_frame)
            if grid_tracker.show_debug:
                grid_tracker.show_debug_images("Debug")

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                logger.info("Processing stopped by user")
                break
            elif key == ord("n"):
                continue
            elif key == ord("d"):
                grid_tracker.toggle_debug_display()
                if grid_tracker.show_debug:
                    grid_tracker.show_debug_images("Debug")
                else:
                    try:
                        if grid_tracker.debug_window_name:
                            cv2.destroyWindow(grid_tracker.debug_window_name)
                            grid_tracker.debug_window_name = None
                    except Exception:
                        pass
                continue

    if show:
        cv2.destroyAllWindows()

    logger.info(f"Processed {len(image_files)} images. Results saved to: {output_path}")

    if gt_annotations and eval_funcs and eval_funcs.get("evaluate_images"):
        try:
            logger.info("Running full evaluation with GT annotations...")
            eval_output_folder = str(output_path / "evaluation_results")
            metrics = eval_funcs["evaluate_images"](
                input_folder=str(input_path),
                annotations_path=gt_annotations_path,
                output_folder=eval_output_folder,
            )
            logger.info(f"Evaluation results saved to: {eval_output_folder}")

            line_metrics = metrics["line_distance_metrics"]
            class_metrics = metrics["classification_metrics"]
            logger.info("=" * 60)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info("Line Distance Metrics:")
            logger.info(f"  Mean: {line_metrics['mean']:.2f}px")
            logger.info(f"  Median: {line_metrics['median']:.2f}px")
            logger.info(f"  Std Dev: {line_metrics['std']:.2f}px")
            logger.info("Classification Metrics:")
            logger.info(f"  Accuracy: {class_metrics['accuracy']*100:.2f}%")
            logger.info(
                f"  Correct: {class_metrics['num_correct']}/"
                f"{class_metrics['num_classifications']}"
            )
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to run full evaluation: {e}", exc_info=True)
    else:
        if not gt_annotations_path:
            logger.info(
                "Evaluation skipped: No GT annotations. Use --gt-annotations to enable."
            )
        elif not gt_annotations:
            logger.info(
                "Evaluation skipped: GT annotations could not be loaded."
            )
        elif not eval_funcs or not eval_funcs.get("evaluate_images"):
            logger.info(
                "Evaluation skipped: evaluate_lid_detection.py not found "
                "or evaluate_images not available."
            )

    if len(all_vials_times) > 0:
        avg_vials = np.mean(all_vials_times)
        avg_grid = np.mean(all_grid_times)
        avg_total = np.mean(all_total_times)
        min_total = np.min(all_total_times)
        max_total = np.max(all_total_times)
        fps_avg = 1000.0 / (avg_total * 1000) if avg_total > 0 else 0
        fps_min = 1000.0 / (max_total * 1000) if max_total > 0 else 0
        fps_max = 1000.0 / (min_total * 1000) if min_total > 0 else 0
        fps_vials = 1000.0 / (avg_vials * 1000) if avg_vials > 0 else 0
        fps_grid = 1000.0 / (avg_grid * 1000) if avg_grid > 0 else 0

        logger.info("=" * 60)
        logger.info("PROFILING SUMMARY - GridTracker v11 (All Images)")
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
        logger.info("Percentage breakdown:")
        logger.info(f"  - Vials Detection: {avg_vials/avg_total*100:.1f}%")
        logger.info(f"  - Grid Generation: {avg_grid/avg_total*100:.1f}%")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize images with WellRack GridTracker v11 and run evaluation."
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Path to folder containing images (default: assets/testing_images).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output folder.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open OpenCV window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--debug-output-folder",
        type=str,
        default=None,
        help="Folder to save debug images.",
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        help="Enable debug display mode.",
    )
    parser.add_argument(
        "--gt-annotations",
        type=str,
        default=None,
        help="Path to GT annotations JSON (lid_annotations.json) for evaluation.",
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
        help="Reference state for D1 (default: CLOSED_LID).",
    )
    parser.add_argument(
        "--no-fft-features",
        action="store_true",
        help="Disable FFT texture features (Fix B).",
    )
    parser.add_argument(
        "--no-pairwise-clustering",
        action="store_true",
        help="Disable pairwise K-Means clustering (Fix C).",
    )
    parser.add_argument(
        "--lid-classifier-profile-sigma-factor",
        type=float,
        default=0.20,
        dest="lid_classifier_profile_sigma_factor",
        help="Gaussian profile sigma factor (Fix E, default: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    default_image_folder = str(Path(__file__).parent / "assets" / "testing_images")
    image_folder = args.image_folder or default_image_folder

    if not Path(image_folder).exists():
        logger.warning(
            f"Image folder does not exist: {image_folder}. "
            "Create it or use --image-folder."
        )
        sys.exit(1)

    process_images_from_folder(
        input_folder=image_folder,
        output_folder=args.output,
        show=not args.no_show,
        debug_output_folder=args.debug_output_folder,
        enable_debug=args.enable_debug,
        gt_annotations_path=args.gt_annotations,
        enable_lid_classification=not args.no_lid_classification,
        lid_classifier_strip_width=args.lid_classifier_strip_width,
        lid_classifier_buffer_size=args.lid_classifier_buffer_size,
        lid_classifier_vial_count_threshold=args.lid_classifier_vial_count_threshold,
        lid_classifier_reference_state=args.lid_classifier_reference_state,
        lid_classifier_use_fft_features=not args.no_fft_features,
        lid_classifier_use_pairwise_clustering=not args.no_pairwise_clustering,
        lid_classifier_profile_sigma_factor=args.lid_classifier_profile_sigma_factor,
    )


if __name__ == "__main__":
    main()
