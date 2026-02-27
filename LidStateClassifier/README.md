# LidStateClassifier

Deterministic classifier that decides, for each column of a vial rack (D1…D8),
whether its lid is **OPEN** or **CLOSED**.  Two classification methods are
available, selected at construction time via `use_fft_features` and
`use_pairwise_clustering`.

---

## Method 1 — Reference-Subtraction (v10)

Used by `visualize_with_gridtracker10_video.py`.

### Concept

D1 (the leftmost detected column) is treated as a fixed reference whose lid
state is known in advance (`reference_state`, default `"CLOSED_LID"`).  Every
other column is compared to D1 by subtracting their image strips and measuring
the mean absolute difference (MAD).  A column that looks similar to D1 is
assigned the same state; one that looks different is assigned the opposite state.

### Pipeline

```
for each frame:
  1. Assign vial centroids to columns via nearest-line distance.
  2. Extract a tight vertical strip (strip_width px wide) for each column,
     convert to greyscale, apply CLAHE normalisation.
  3. Compute MAD(D1_strip, Di_strip) for every column i ≠ D1.
  4. Gate 1 – structural check (hard override):
       if vial_count(Di) / vial_count(D1)  ≤  vial_count_threshold
           → OPEN_LID  (too few vials detected → column is likely open)
  5. Gate 2 – visual similarity:
       compute adaptive visual_threshold from the distribution of
       {MAD(D1,D2), …, MAD(D1,DN)} using diff_threshold_method:
           "percentile"  → 60th percentile of the diff array
           "median_iqr"  → median + iqr_multiplier × IQR
           "fixed"       → user-supplied diff_threshold value
       apply min_visual_threshold floor to prevent collapse at small N.
       if MAD(D1,Di) < visual_threshold  → same state as D1
       else                              → opposite state to D1
  6. Temporal smoothing: majority vote over the last buffer_size calls.
```

### Key parameters (v10 defaults)

| Parameter | Default | Effect |
|---|---|---|
| `strip_width` | 40 px | Width of per-column crop |
| `buffer_size` | 5 | Majority-vote window |
| `vial_count_threshold` | 0.5 | Gate 1 ratio cutoff |
| `diff_threshold_method` | `"percentile"` | Threshold computation strategy |
| `min_visual_threshold` | 0.0 (off) | Floor for adaptive threshold |
| `reference_state` | `"CLOSED_LID"` | Assumed state of D1 |
| `use_fft_features` | `False` | Disabled — uses raw MAD |
| `use_pairwise_clustering` | `False` | Disabled — uses D1 reference |

### Known limitations

- **D1 assumption is fragile.** If the leftmost column ever changes state, every
  other classification inverts silently.
- **Per-frame threshold is statistically weak.** With N = 7 differences the
  percentile/IQR estimate is noisy; a single specular reflection shifts the
  threshold enough to flip all outputs.
- **Raw MAD is sensitive to local reflections** on glass vials and to stale
  centroid positions between grid updates.

---

## Method 2 — FFT Texture + Gap-Based Split (v11)

Used by `visualize_with_gridtracker11_video.py`.

### Concept

Instead of comparing every column to a fixed D1 reference, each column is
characterised independently by the **frequency-domain texture** of its strip.
Open wells expose a periodic ring structure (dark interior, bright rim) that
produces high power at mid-to-high spatial frequencies.  Closed lids are flat
and nearly uniform, producing low frequency power.  A 1-D gap search on the
sorted feature values then decides whether a genuine state boundary exists
without ever assuming D1's state.

### Pipeline

```
for each frame:
  1–2. Same strip extraction as Method 1 (assign vials, CLAHE crop).
  3. FFT texture feature per column (Fix E applied here):
       if profile_sigma_factor > 0:
           profile = strip @ gaussian_weights(strip_width, sigma=strip_width*σ)
       else:
           profile = strip.mean(axis=1)        # plain box average
       profile -= profile.mean()               # remove DC / mean brightness
       spectrum = |FFT(profile)|²
       feature  = mean(spectrum[low_cut:high_cut]) / mean(spectrum[1:])
                  where low_cut = n//8,  high_cut = n*3//4
       Result: high → textured (open well), low → flat (closed lid).
  4. Gap-based state split (replaces K-Means):
       sort columns by ascending feature value.
       gaps = diff(sorted_features)            # N-1 gaps between neighbours
       max_gap = max(gaps)
       if max_gap < min_feature_separation (0.08):
           no meaningful boundary found → fall back to Method 1 reference gate.
       else:
           split at the largest gap:
               columns below gap → CLOSED_LID
               columns above gap → OPEN_LID
  5. Gate 1 structural override still applies (same as Method 1).
  6. Temporal smoothing: majority vote over buffer_size calls, but
     classification runs every frame (Fix D) — not only on grid-update frames —
     so the buffer receives 5× more observations than in v10.
```

### Why gap-based instead of K-Means

K-Means with k = 2 always forces a binary split.  When all columns are in the
same state, natural inter-column FFT variation (caused by illumination gradients,
vial density, perspective) can easily span 0.06–0.10 — enough to pass a simple
range check, yet not a real state boundary.  K-Means would then output, e.g.,
D1 = OPEN and D2–D4 = CLOSED even with a fully uncovered rack.

The gap check looks for a **discontinuity**, not just spread.  A genuine
open/closed boundary produces a gap of 0.15–0.25 between the two groups.
Same-state natural variation produces gaps of 0.01–0.06.  A threshold of 0.08
cleanly separates them.

```
All OPEN  [0.20, 0.22, 0.28, 0.30]  gaps [0.02, 0.06, 0.02]  max=0.06 < 0.08 → fallback
All CLOSED [0.07, 0.09, 0.12, 0.14] gaps [0.02, 0.03, 0.02]  max=0.03 < 0.08 → fallback
1 open    [0.08, 0.09, 0.10, 0.32]  gaps [0.01, 0.01, 0.22]  max=0.22 > 0.08 → SPLIT ✓
2 open    [0.08, 0.10, 0.28, 0.31]  gaps [0.02, 0.18, 0.03]  max=0.18 > 0.08 → SPLIT ✓
```

When the gap check falls back to Method 1, the reference-subtraction gate
handles the "all same state" case; Method 1 is correct in this regime because
all columns look alike regardless of D1's assumption.

### Fix E — Gaussian-weighted strip profile

The grid-tracker occasionally positions the **leftmost column line** (D1) within
20 px of the rack's internal cell-wall divider (the green plastic grid structure
that separates adjacent vial columns).  The 40 px strip therefore captures part
of that divider, which creates high-contrast rectangular features whose FFT power
is much larger than the vial-cap pattern of interior columns.  Without correction
the gap-split fires spuriously, classifying D1 as OPEN even when all lids are
clearly closed.

**Fix:** apply a Gaussian kernel of width `strip_width × profile_sigma_factor`
across the strip *before* averaging to the 1D vertical profile.  This
de-emphasises edge pixels (where the cell-wall contaminates the strip) and
focuses the profile on the column's centre — where the actual vial caps sit.
Empirical validation on the reference video shows that `sigma_factor = 0.20`
reduces spurious gaps from 0.20–0.27 (all-closed, without fix) to 0.04–0.06
(below the 0.08 threshold) while preserving large gaps (0.17–0.33) when
columns are genuinely in different states.

### Key parameters (v11 defaults)

| Parameter | Default | Effect |
|---|---|---|
| `strip_width` | 40 px | Width of per-column crop |
| `buffer_size` | 5 | Majority-vote window |
| `vial_count_threshold` | 0.20 | Gate 1 ratio cutoff (tightened) |
| `diff_threshold_method` | `"median_iqr"` | Fallback threshold method |
| `iqr_multiplier` | 0.8 | IQR weight for fallback threshold |
| `min_visual_threshold` | 40.0 | Floor for fallback threshold |
| `reference_state` | `"CLOSED_LID"` | Used only in fallback path |
| `use_fft_features` | `True` | Enable FFT texture feature |
| `use_pairwise_clustering` | `True` | Enable gap-based split |
| `min_feature_separation` | 0.08 | Minimum gap to accept a split |
| `profile_sigma_factor` | 0.20 | Gaussian edge-suppression (Fix E) |

---

## Constructor

```python
LidStateClassifier(
    strip_width=40,
    buffer_size=5,
    vial_count_threshold=0.5,
    diff_threshold=None,
    diff_threshold_method="percentile",
    iqr_multiplier=1.0,
    min_visual_threshold=0.0,
    reference_state="CLOSED_LID",
    use_fft_features=False,        # True → Method 2 primary path
    use_pairwise_clustering=False,  # True → gap-based split
    min_feature_separation=0.08,
    profile_sigma_factor=0.0,      # 0.20 → Fix E (Gaussian edge-suppression)
    debug=False,
)
```

`use_fft_features=False` (default) reproduces Method 1 exactly.
Set both `use_fft_features=True` and `use_pairwise_clustering=True` to
activate Method 2 with Method 1 as its automatic fallback.
Set `profile_sigma_factor=0.20` (v11 default) to apply Fix E.

## Output

```python
states = classifier.classify(frame, dominant_lines, centroids)
# {"D1": "CLOSED_LID", "D2": "OPEN_LID", "D3": "CLOSED_LID", ...}
```
