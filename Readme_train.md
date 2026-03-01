# Vial Keypoint Detection Model -- Training Report

## Overview

A heatmap regression model for detecting vial centre positions in laboratory
rack images.  The model replaces the existing HSL + KMeans + connected-components
pipeline with a learned approach that handles both open and closed lids.

**Architecture:** Simple Baselines (Xiao et al., 2018) --
ResNet-18 backbone + 3 transposed-conv upsampling layers + 1x1 conv head.

```
Input (B, 3, 384, 384)
       |
ResNet-18 (pretrained ImageNet)  -->  (B, 512, 12, 12)
       |
3x DeconvBlock (512->256->128->64)  -->  (B, 64, 96, 96)
       |
1x1 Conv  -->  (B, 1, 96, 96)   heatmap  (output stride = 4)
```

Total parameters: **13,929,985** (all trainable).

---

## Dataset

| Split | Images | Source |
|-------|--------|--------|
| Train | 56     | `assets/testing_images/` |
| Val   | 6      | 10% of train split (random) |
| Test  | 6      | Held out -- `img_1.png`, `img_10.png`, `img_2.png`, `img_14.png`, `img_43.jpg`, `img_7.png` |

Test images were selected for diversity:
- 2 closed-lid only (large and medium resolution)
- 4 mixed open+closed lids (large, medium, small, very-small portrait)

**Annotations:** Pseudo-keypoints generated from existing row-line annotations
by interpolating 12 equally-spaced positions per row (48 keypoints/image for
4-row racks).  Stored in `assets/testing_images/vial_keypoints.json`.

Splits are recorded in `assets/testing_images/splits.json`.

---

## Augmentation Pipeline

Applied during training (each independently with its own probability):

| Augmentation | Probability | Range |
|--------------|-------------|-------|
| Horizontal flip | 0.50 | -- |
| Vertical flip | 0.50 | -- |
| Rotation | 0.50 | +/-15 deg |
| Scale jitter | 0.50 | 0.8x -- 1.2x |
| Translation shift | 0.30 | +/-5% |
| Brightness jitter | 0.40 | +/-40 intensity |
| Contrast jitter | 0.30 | 0.7x -- 1.3x |
| HSV saturation/value | 0.30 | S: 0.6-1.4x, V: 0.7-1.3x |
| Gaussian noise | 0.25 | sigma 3--15 |
| Gaussian blur | 0.20 | kernel 3 or 5 |

---

## Training Configuration

```
backbone:       resnet18  (ImageNet pretrained)
input_size:     384 x 384
heatmap_stride: 4         (output: 96 x 96)
heatmap_sigma:  2.5       (Gaussian blob radius in heatmap pixels)
batch_size:     4
optimizer:      Adam (lr=1e-3, weight_decay=1e-4)
lr_schedule:    MultiStepLR  milestones=[60, 100, 130]  gamma=0.1
epochs:         150
loss:           MSE on heatmap
seed:           42
device:         CPU (4 cores, 15 GB RAM)
```

---

## Training Log

```
Epoch 001/150 | train_loss=0.141221 | val_loss=0.133002 | P=0.247 R=0.872 dist=2.45 | lr=1.0e-03
Epoch 002/150 | train_loss=0.043270 | val_loss=0.075366 | P=0.559 R=0.615 dist=2.25 | lr=1.0e-03
Epoch 003/150 | train_loss=0.028961 | val_loss=0.029322 | P=0.770 R=0.701 dist=2.16 | lr=1.0e-03
Epoch 004/150 | train_loss=0.022766 | val_loss=0.018408 | P=0.704 R=0.750 dist=1.99 | lr=1.0e-03
Epoch 005/150 | train_loss=0.021442 | val_loss=0.020400 | P=0.421 R=0.788 dist=1.97 | lr=1.0e-03
Epoch 010/150 | train_loss=0.015127 | val_loss=0.016752 | P=0.379 R=0.767 dist=2.05 | lr=1.0e-03
Epoch 020/150 | train_loss=0.015888 | val_loss=0.014020 | P=0.639 R=0.837 dist=2.01 | lr=1.0e-03
Epoch 030/150 | train_loss=0.013837 | val_loss=0.022814 | P=0.578 R=0.712 dist=2.24 | lr=1.0e-03
Epoch 040/150 | train_loss=0.013823 | val_loss=0.017901 | P=0.538 R=0.830 dist=2.12 | lr=1.0e-03
Epoch 050/150 | train_loss=0.015041 | val_loss=0.018952 | P=0.405 R=0.653 dist=2.00 | lr=1.0e-03
Epoch 060/150 | train_loss=0.013732 | val_loss=0.013793 | P=0.528 R=0.802 dist=2.08 | lr=1.0e-04
Epoch 070/150 | train_loss=0.010461 | val_loss=0.012057 | P=0.571 R=0.764 dist=1.85 | lr=1.0e-04
Epoch 080/150 | train_loss=0.010339 | val_loss=0.011388 | P=0.550 R=0.767 dist=1.82 | lr=1.0e-04
Epoch 090/150 | train_loss=0.009789 | val_loss=0.010781 | P=0.552 R=0.833 dist=1.56 | lr=1.0e-04
Epoch 100/150 | train_loss=0.009003 | val_loss=0.011544 | P=0.583 R=0.917 dist=1.42 | lr=1.0e-05
Epoch 110/150 | train_loss=0.008161 | val_loss=0.009706 | P=0.575 R=0.913 dist=1.28 | lr=1.0e-05
Epoch 120/150 | train_loss=0.007980 | val_loss=0.010209 | P=0.591 R=0.917 dist=1.33 | lr=1.0e-05
Epoch 130/150 | train_loss=0.007518 | val_loss=0.009849 | P=0.610 R=0.920 dist=1.29 | lr=1.0e-06
Epoch 140/150 | train_loss=0.007107 | val_loss=0.009720 | P=0.607 R=0.910 dist=1.26 | lr=1.0e-06
Epoch 150/150 | train_loss=0.007091 | val_loss=0.009477 | P=0.621 R=0.931 dist=1.24 | lr=1.0e-06
```

**Best checkpoint:** epoch 122, val_loss = 0.009270

**Total training time:** 20.3 minutes (CPU)

### Convergence Notes

- Loss drops rapidly in the first 5 epochs (0.14 -> 0.02)
- LR decay at epoch 60 (1e-3 -> 1e-4) yields a clear improvement in val_loss
- Second LR decay at epoch 100 (-> 1e-5) further refines localisation (dist drops from 1.82 to 1.28)
- Recall on val set reaches 93% by epoch 150; precision sits at ~62%
- The moderate precision on the val set is due to the strict 4-pixel matching threshold in heatmap space (= 16px in image space); on the actual test set with the more lenient original-space matching, precision reaches 99%

---

## Test Set Evaluation Results

Evaluated using `best_model.pth` (epoch 122) with threshold=0.3, NMS radius=5.

### Per-Image Results

| Image | Size | Lid Type | Pred | GT | TP | FP | FN | Precision | Recall | F1 | Dist (px) |
|-------|------|----------|------|----|----|----|-----|-----------|--------|-----|-----------|
| img_1.png | 883x646 | closed | 46 | 48 | 44 | 2 | 4 | 0.957 | 0.917 | 0.936 | 11.7 |
| img_10.png | 872x553 | closed | 48 | 48 | 48 | 0 | 0 | 1.000 | 1.000 | 1.000 | 11.1 |
| img_2.png | 1067x667 | mixed | 48 | 48 | 48 | 0 | 0 | 1.000 | 1.000 | 1.000 | 6.6 |
| img_14.png | 398x256 | mixed | 47 | 48 | 47 | 0 | 1 | 1.000 | 0.979 | 0.989 | 4.1 |
| img_43.jpg | 150x224 | mixed | 36 | 48 | 36 | 0 | 12 | 1.000 | 0.750 | 0.857 | 3.8 |
| img_7.png | 892x575 | mixed | 45 | 48 | 45 | 0 | 3 | 1.000 | 0.938 | 0.968 | 10.3 |

### Aggregate Metrics

| Metric | Macro-avg | Micro-avg |
|--------|-----------|-----------|
| **Precision** | 0.993 | 0.993 |
| **Recall** | 0.931 | 0.931 |
| **F1** | 0.958 | 0.961 |
| **Mean dist** | 7.9 px | -- |

**Total: 268 TP, 2 FP, 20 FN** across 288 ground-truth keypoints.

### Per-Image Analysis

- **img_10.png, img_2.png**: Perfect detection -- 48/48 with zero false positives.
- **img_14.png**: Near-perfect (47/48), missed one edge vial.
- **img_1.png**: 44/48 detected with 2 FP; the 4 missed are likely at rack edges.
- **img_7.png**: 45/48, strong performance on mixed open+closed rack.
- **img_43.jpg**: Lowest recall (75%). This is the smallest image (150x224 portrait), where 12 vials are missed. Upscaling from 150px to 384px causes significant blur, degrading small-feature detection.

### Mean Distance Error

The mean distance between matched predictions and GT ranges from 3.8px to 11.7px
in original image space. Larger images have proportionally larger pixel distances
because the heatmap resolution (96x96) is fixed; the mapping back to a 900px-wide
image amplifies sub-pixel heatmap errors.

---

## Output Files

| Path | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Best checkpoint (epoch 122, val_loss=0.009270) |
| `checkpoints/last_model.pth` | Final checkpoint (epoch 150) |
| `checkpoints/epoch_50.pth` | Epoch 50 checkpoint |
| `checkpoints/epoch_100.pth` | Epoch 100 checkpoint |
| `checkpoints/epoch_150.pth` | Epoch 150 checkpoint |
| `checkpoints/history.json` | Full training history (per-epoch metrics) |
| `checkpoints/train_log.txt` | Raw training log |
| `eval_results/eval_summary.json` | Test-set evaluation metrics |
| `eval_results/*_eval.png` | Side-by-side visualisation (detections + heatmap overlay) |

---

## How to Reproduce

```bash
# Install dependencies
pip install torch torchvision opencv-python-headless numpy

# Train
python -m keypoint_model.train \
    --annotations assets/testing_images/vial_keypoints.json \
    --images assets/testing_images \
    --splits assets/testing_images/splits.json \
    --epochs 150 --batch-size 4 --lr 1e-3 \
    --lr-steps 60 100 130 --lr-factor 0.1 \
    --backbone resnet18 --input-h 384 --input-w 384 \
    --sigma 2.5 --seed 42 --save-dir checkpoints

# Evaluate
python -m keypoint_model.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --annotations assets/testing_images/vial_keypoints.json \
    --images assets/testing_images \
    --splits assets/testing_images/splits.json \
    --output-dir eval_results

# Single-image inference
python -m keypoint_model.predict \
    --checkpoint checkpoints/best_model.pth \
    --image assets/testing_images/img_1.png \
    --output prediction.png
```

---

## Potential Improvements

1. **Manual annotation refinement** -- The current keypoints are pseudo-labels
   interpolated uniformly along row lines. Refining with the annotation tool
   (`python -m keypoint_model.annotate_tool`) would improve GT accuracy
   and therefore model precision.

2. **More training data** -- 56 training images is small. Adding more rack images
   from video frames would improve generalisation, especially for edge cases
   like very small crops and unusual viewpoints.

3. **Higher input resolution** -- Using 512x512 instead of 384x384 would improve
   detection on large images (reducing heatmap quantisation error) at the cost of
   ~2x training time.

4. **GPU training** -- The current run was CPU-only (20 min). With a GPU, training
   could be extended to 300+ epochs with larger input in minutes.

5. **FPN / U-Net decoder** -- Adding skip connections from earlier ResNet layers
   would improve spatial precision and small-object detection (relevant for
   img_43.jpg-type cases).
