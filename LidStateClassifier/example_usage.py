"""
Example usage of LidStateClassifier.

This script demonstrates how to use the LidStateClassifier with the
GridTracker pipeline.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple

from V2.GridTracker.LidStateClassifier import LidStateClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic example of using LidStateClassifier."""
    
    # Initialize classifier
    classifier = LidStateClassifier(
        patch_size=40,
        buffer_size=5,
        threshold=None,  # Use Otsu thresholding
        use_otsu=True,
        debug=False,
    )
    
    # Example frame (BGR image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Example dominant lines: ((angle_deg, d), count, line_id)
    # Format: angle in degrees, d is distance parameter
    dominant_lines = [
        ((90.0, 100.0), 1, "D1"),  # Vertical line at x=100
        ((90.0, 200.0), 1, "D2"),  # Vertical line at x=200
        ((90.0, 300.0), 1, "D3"),  # Vertical line at x=300
    ]
    
    # Example vial centroids
    centroids = [
        (100, 100),  # Near D1
        (100, 200),  # Near D1
        (200, 150),  # Near D2
        (200, 250),  # Near D2
        (300, 120),  # Near D3
    ]
    
    # Classify lid states
    column_states = classifier.classify(frame, dominant_lines, centroids)
    
    logger.info(f"Classification results: {column_states}")
    
    # Get column scores (for debugging)
    if classifier.debug:
        scores = classifier.get_column_scores()
        logger.info(f"Column scores: {scores}")


def example_with_custom_threshold():
    """Example using a custom threshold instead of Otsu."""
    
    classifier = LidStateClassifier(
        patch_size=40,
        buffer_size=5,
        threshold=0.4,  # Custom threshold
        use_otsu=False,
        debug=True,
    )
    
    # ... rest of usage is the same
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dominant_lines = [((90.0, 100.0), 1, "D1")]
    centroids = [(100, 100)]
    
    column_states = classifier.classify(frame, dominant_lines, centroids)
    logger.info(f"Classification with custom threshold: {column_states}")


def example_integration_with_gridtracker():
    """
    Example showing integration with GridTrackerWithExternalMask.
    
    This is how the classifier is used in the actual pipeline.
    """
    from gridtracker_functions_method_best import VialsDetector
    
    # Initialize detector
    detector = VialsDetector()
    
    # Initialize classifier
    classifier = LidStateClassifier()
    
    # Process a frame
    frame = cv2.imread("example_frame.jpg")  # Load your frame
    
    # Set plate mask (from YOLO detection or manual annotation)
    plate_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    # ... fill in plate_mask with actual detection
    
    detector.set_plate_mask(plate_mask)
    
    # Detect vials and get grid
    bboxes, centroids, vis_frame, debug_images, overlay, filtered_lines, points, inferred = (
        detector.detect_and_track_vials(frame)
    )
    
    # Extract dominant lines
    dominant_lines = [
        (line[0], line[1], f"D{i+1}")
        for i, line in enumerate(filtered_lines)
        if len(line) >= 3 and line[2]  # is_dominant flag
    ]
    
    # Classify lid states
    column_states = classifier.classify(frame, dominant_lines, centroids)
    
    logger.info(f"Lid states: {column_states}")
    
    return column_states


if __name__ == "__main__":
    logger.info("Running basic usage example...")
    example_basic_usage()
    
    logger.info("\nRunning custom threshold example...")
    example_with_custom_threshold()
