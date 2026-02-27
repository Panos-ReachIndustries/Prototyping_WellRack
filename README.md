# GridTracker

## Purpose

The GridTracker module provides real-time grid detection and tracking capabilities for laboratory automation applications, particularly focused on vial rack management. It uses YOLO-based object detection combined with advanced computer vision techniques to identify and track grid structures in video frames. The module employs line detection algorithms, morphological operations, and confidence-based tracking to maintain stable grid references across frame sequences, enabling precise cell location identification and visual annotation of grid structures for automated laboratory workflows.

## Weight Files

| File Name | Purpose | Storage Location |
|-----------|---------|------------------|
| yolo_v11l_multiclass.pt | YOLO model for pipette and plate detection in laboratory environments |  [S3](https://eu-west-2.console.aws.amazon.com/s3/object/reach-ml-weights?region=eu-west-2&bucketType=general&prefix=3d-detection/yolo_v11n_3d.pt) - place in `/weights/yolo_best_v11n.pt` |

## Function Definitions

### GridTracker.__init__()
Initializes the GridTracker with YOLO model loading, FPS tracking, and internal state management for grid tracking.

**Inputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| None | - | No input parameters required |

**Outputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| None | - | Initializes internal state (dominant_lines, recessive_lines, vials_grid_buffer, detector, pipette_model) |

### GridTracker.track_frame(frame)
Processes a single frame to detect and update grid tracking information using YOLO detection and line analysis algorithms.

**Inputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| frame | numpy.ndarray[uint8] | Input video frame in BGR format for grid detection and tracking |

**Outputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| None | - | Updates internal grid state (dominant_lines, recessive_lines, current_grid_points, best_vials_entry) |

### GridTracker.draw_frame(frame)
Annotates the input frame with current grid tracking information including dominant/recessive lines, confidence scores, and FPS metrics.

**Inputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| frame | numpy.ndarray[uint8] | Input video frame in BGR format to be annotated |

**Outputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| annotated_frame | numpy.ndarray[uint8] | Frame with grid visualization overlay including colored lines, confidence scores, FPS counter, and frame number |

### GridTracker.get_grid()
Returns the current grid tracking information including line parameters, intersection points, and confidence metrics.

**Inputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| None | - | No input parameters required |

**Outputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| grid_info | dict[str, Any] | Dictionary containing grid tracking data |
| grid_info.dominantLines | list[tuple[float, float, str]] | List of dominant grid lines with (angle_deg, distance, line_id) |
| grid_info.recessiveLines | list[tuple[float, float, str]] | List of recessive grid lines with (angle_deg, distance, line_id) |
| grid_info.points | list[tuple[float, float]] | List of detected grid intersection points as (x, y) coordinates |
| grid_info.frameCount | int | Current frame number being processed |
| grid_info.confidence | float | Confidence score of the current best grid detection |

### GridTracker.get_cell_reference(x, y)
Finds the closest row and column identifiers for given pixel coordinates using distance calculations to grid lines.

**Inputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| x | float | X pixel coordinate in the image frame |
| y | float | Y pixel coordinate in the image frame |

**Outputs:**
| Arg Name/Output Name | Type | Description |
|---------------------|------|-------------|
| cell_reference | tuple[str, str] | Tuple containing (row_id, col_id) of the closest grid cell |

## Limitations

- Requires well-defined grid structures with clear line patterns
- Performance depends on lighting conditions and image quality
- YOLO model requires appropriate training data for specific laboratory equipment
- Grid tracking may fail with highly distorted or occluded rack views 
