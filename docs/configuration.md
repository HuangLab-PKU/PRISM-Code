# Configuration Guide

## Overview

PRISM uses YAML configuration files for the **readout** and **gene calling** stages. **Cell segmentation** is configured in Python (no YAML). Configuration files live in `configs/`:

```
configs/
├── readout.yaml                  # Spot detection + intensity readout
├── gene_calling.yaml             # Gene-calling loader (selects modules + overrides)
├── gene_calling_base.yaml        # Shared gene-calling params (channel correction, codebook, panel, features)
├── gene_calling_gmm.yaml         # GMM method parameters
├── gene_calling_full.yaml        # Variant preset
├── gene_calling_spherical.yaml   # Variant preset (spherical covariance)
└── codebooks/
    └── prism30.csv               # PRISM30 codebook
```

> Image-processing configuration (stitching, registration, illumination correction) lives in the companion `spatial_img_core` package, not in this repo.

---

## 1. Readout Configuration (`readout.yaml`)

Used by `scripts/readout.py`. The `RUN_ID` is passed on the command line; everything else comes from this file.

```yaml
# Base directory for processed data; outputs go to BASE_DIR / "{RUN_ID}_processed"
base_dir: "/path/to/processed"

# Stitched input filenames (under stitched/); the stem becomes the channel label in output
channel_files:
  - cyc_1_cy5.tif      # Cy5,   excitation 628 nm
  - cyc_1_TxRed.tif    # TxRed, excitation 586 nm
  - cyc_1_cy3.tif      # Cy3,   excitation 531 nm
  - cyc_1_FAM.tif      # FAM,   excitation 485 nm

# Spot detection
detection_method: spotiflow   # spotiflow (default) | gaussian_tophat | tophat
detection_snr:                # per-file-stem SNR for the traditional methods
  cyc_1_cy5: 3.0
  cyc_1_TxRed: 3.0
  cyc_1_cy3: 3.0
  cyc_1_FAM: 3.0

# Intensity reading (tophat + local max)
tophat_radius: 3
search_radius: 1              # 3x3 max

# Deduplication
dedup_threshold: 2           # pixels

# Block processing
block_size: [2048, 2048]
block_overlap: [64, 64]

# Parallel workers
n_workers: 4
```

**Parameter notes**:
- `base_dir`: root that contains `<RUN_ID>_processed/`. Edit this for your machine.
- `channel_files`: one stitched TIFF per channel; the filename stem (e.g. `cyc_1_cy5`) is used as the column name in `intensity.csv`.
- `detection_method`: `spotiflow` is the deep-learning default (the `readout.py` spotiflow path uses `device='cuda'`, so it needs a GPU). `gaussian_tophat` / `tophat` are CPU-friendly traditional fallbacks that use `detection_snr` and `tophat_radius`.
- `block_size` / `block_overlap`: large stitched images are processed block-by-block; overlap avoids edge effects. `n_workers` sets process-pool parallelism.

---

## 2. Gene Calling Configuration (modular)

Gene calling is config-driven and modular. `gene_calling.yaml` lists the modules to load and any overrides:

```yaml
# gene_calling.yaml
config_modules:
  - gene_calling_base.yaml
  - gene_calling_gmm.yaml

overrides:
  # e.g. classification:
  #        gmm:
  #          covariance_type: spherical
```

Variant presets such as `gene_calling_spherical.yaml` and `gene_calling_full.yaml` simply load the same base/method modules with different `overrides`.

### Method selection

The classification method is set under `classification.method`:

| Method | Status | Notes |
|---|---|---|
| `gmm` | **default** | Gaussian Mixture Model decoding (`gene_calling_gmm.yaml`) |
| `codebook_gmm` | supported | GMM constrained by a codebook (`codebook.path`) |
| `postcode` | **experimental** | Probabilistic decoding; opt-in. Requires the vendored PoSTcode package and a `postcode:` config section. See the [Installation Guide](installation.md). |

### Channel correction (`gene_calling_base.yaml`)

Maps `intensity.csv` columns (excitation wavelength) to fluorophore names and applies a 4×4 linear unmixing matrix to remove spectral crosstalk / FRET. The default is the identity matrix (no correction); use `scripts/calibrate_channels.py` to estimate the matrix from calibration experiments.

```yaml
channel_correction:
  channels:
    r01_ex628: {fluorophore: r01_Cy5,   excitation_nm: 628, emission_nm: 670, role: color}
    r01_ex586: {fluorophore: r01_TxRed, excitation_nm: 586, emission_nm: 615, role: color}
    r01_ex531: {fluorophore: r01_Cy3,   excitation_nm: 531, emission_nm: 550, role: layer}
    r01_ex485: {fluorophore: r01_FAM,   excitation_nm: 485, emission_nm: 520, role: color}

  # corrected = raw @ M^T  (column order follows channels above)
  transform_matrix:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  clip_negative: true
```

The `role` field distinguishes **color** channels (encode the barcode hue) from the **layer** channel (encodes intensity stratification).

### Codebook & panel

```yaml
codebook:
  path: codebooks/prism30.csv   # relative to the config directory

prism_panel:
  type: "PRISM30"               # PRISM30, PRISM31, PRISM45, PRISM46, PRISM63, PRISM64
  num_per_layer: 15
  g_layer_num: 2
  total_components: 30
  channel_grading:
    color_channels: 5
    layer_channel: 2
```

### GMM parameters (`gene_calling_gmm.yaml`)

```yaml
classification:
  method: "gmm"
  gmm:
    covariance_type: "diag"     # full | tied | diag | spherical
    use_4d_ratios_only: true
    max_iter: 100
    tol: 1e-3
    random_state: 42
    initialization:
      method: "kmeans"
      n_init: 10
```

See `gene_calling_base.yaml` for the full set of feature-extraction, evaluation, and visualization options.

---

## 3. Cell Segmentation Configuration

Cell segmentation has **no YAML file**. Parameters are defined by the `SegmentationConfig` class in `prism/cell_segmentation/segmentation_config.py`, and the common ones are exposed as `scripts/segment_dapi.py` command-line flags:

```bash
python scripts/segment_dapi.py <RUN_ID> \
    --base-dir /path/to/processed \
    --nucleus-image cyc_1_DAPI.tif \
    --method auto \           # watershed | stardist | auto
    --dimension auto          # 2d | 3d | auto
```

`SegmentationConfig` holds `watershed_params`, `stardist_2d_params`, `stardist_3d_params`, and `general_params`. To change advanced parameters (block size, minimum cell size, StarDist model name, etc.), edit the class or construct it in your own script.

---

## Configuration Usage

The gene-calling pipeline loads and merges modular configs for you:

```python
from prism.gene_calling.pipeline import SignalClassificationPipeline

# Loads gene_calling.yaml, merges its config_modules + overrides
pipeline = SignalClassificationPipeline(config_path="configs/gene_calling.yaml")
```

The readout script loads `readout.yaml` directly from its `--config-dir` (default `configs/`).

---

## Best Practices

- **Version-control your configs** and document parameter changes in the [changelog](changelog.md).
- **Start from the defaults**, then tune on a small crop before processing a full run.
- For poor spot detection, adjust `detection_method` / `detection_snr` / `tophat_radius` in `readout.yaml`.
- For channel crosstalk, re-estimate `transform_matrix` with `scripts/calibrate_channels.py` rather than hand-editing.
- For memory pressure on large images, reduce `block_size` or `n_workers` in `readout.yaml`.
