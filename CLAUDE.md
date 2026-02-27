# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WellRack is a computer vision module within the Lumi-AI-Core monorepo for real-time grid detection and tracking of laboratory vial racks. It combines YOLO-based object detection with custom line detection algorithms (MSER, morphological operations, angle histograms) to identify and track 8x12 grid structures in video frames.

Part of Reach Industries' laboratory automation platform. The parent repo (`Lumi-AI-Core`) contains independent AI/CV modules, each with its own `requirements.txt`.

## Setup and Running

```bash
# Install dependencies
pip install -r requirement.txt

# YOLO weights must be placed at /weights/yolo_best_v11n.pt
# Download from S3: eu-west-2, reach-ml-weights bucket

# Run single-plate video visualization
python visualize_with_gridtracker1_video.py \
    --video path/to/video.mp4 \
    --annotations path/to/annotations.json \
    --output path/to/output.mp4

# Run multi-plate (9) video visualization
python visualize_with_gridtracker9_video.py \
    --video path/to/video.mp4 --output path/to/output.mp4

# Run image batch visualization
python visualize_with_gridtracker9_imgs.py
```

There is no build system, test suite, or Makefile. The module is used via direct import or the visualization scripts.

## Linting

Pre-commit hooks are configured at the repo root (`.pre-commit-config.yaml`):
- **flake8** with `--max-line-length=134`
- **isort** for import sorting
- trailing-whitespace and end-of-file-fixer

## Architecture

### Module Dependency Graph

```
GridTracker.py (main public API)
  ├── gridtracker_functions.py  (VialsDetector class + utilities, aliased as "prototype")
  │   └── grid_lines_functions.py  (line detection, angle computation, line merging)
  └── ultralytics.YOLO  (plate detection model)

visualize_with_gridtracker*.py  (entry-point scripts)
  └── gridtracker_functions.py  (VialsDetector, get_plate_masks)
```

### Core Classes

- **`GridTracker`** (`GridTracker.py`) — Primary API. Manages frame-by-frame grid tracking with YOLO plate detection, a 100-frame buffer for best-detection selection, and confidence scoring.
- **`VialsDetector`** (`gridtracker_functions.py`) — Handles vial detection pipeline: CLAHE preprocessing → MSER/wavelet-based region detection → morphological refinement → centroid extraction → grid line inference.
- **`GridTrackerWithExternalMask`** (`visualize_with_gridtracker1_video.py`) — Extended GridTracker accepting external plate masks or YOLO-World detection with text prompts.
- **`LineTracker`** (`visualize_with_gridtracker1_video.py`) — Assigns persistent spatial IDs to grid lines (D1-D8 for dominant/vertical, R1-R12 for recessive/horizontal).

### Variant Files

`gridtracker_functions_5.py` and `gridtracker_functions_opt.py` are extended/optimized versions of `gridtracker_functions.py` (currently untracked). They represent iterative optimization attempts with the same core API.

### Key Data Representations

**Line normal form:** `(angle_deg, d)` where `x*cos(angle_deg+90°) + y*sin(angle_deg+90°) = d`

**Grid info dict** (from `get_grid()`):
```python
{"dominantLines": [(angle, dist, id), ...],    # vertical lines
 "recessiveLines": [(angle, dist, id), ...],    # horizontal lines
 "points": [(x, y), ...],                       # grid intersections
 "frameCount": int, "confidence": float}
```

**Confidence formula:** `0.4 * (points/96) + 0.3 * (inferred/detected ratio) + 0.3 * (lines/20)`

### Processing Pipeline (track_frame)

1. YOLO detects plate region (class 4) → binary mask
2. Mask eroded (5x5 kernel, 5 iters) and validated against previous frame (IoU > 0.1)
3. VialsDetector runs vial detection (MSER + morphology + connected components)
4. Grid inference: angle histogram → dominant/recessive line detection → line merging (angle threshold 20°, distance threshold 30px) → top 15% filtering
5. Best detection selected from 100-frame buffer by confidence score

## Key Dependencies

- **ultralytics** — YOLO inference for plate/rack detection
- **opencv-python** — Core image processing and morphological operations
- **PyWavelets** — Wavelet decomposition for vial feature extraction
- **scikit-learn** — KMeans clustering for angle/line grouping
- **torch/torchvision** — Neural network inference backend (CUDA if available)

## Git Workflow

- Main integration branch: `dev`
- Current working branch: `test/WellRack_1`
- Feature branches follow `feature/` or `test/` prefix conventions
