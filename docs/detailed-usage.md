# Detailed Usage Guide

This guide walks through the full PRISM **post-stitching** workflow. PRISM starts from stitched per-channel images; the upstream raw-image → stitched-image steps live in the companion `spatial_img_core` package (**not yet public** — request access at **huanglab111@gmail.com**).

All commands run from the repository root after `pip install -e .` (see the [Installation Guide](installation.md)).

## Data Layout

PRISM reads stitched images from `<RUN_ID>_processed/stitched/` and writes results to `readout/`, `segmented/`, and `visualization/` under the same `<RUN_ID>_processed/` directory. See [Data Architecture](data-architecture.md) for the full directory layout, file-naming conventions, and the upstream directories produced by `spatial_img_core`.

A typical stitched input set (one TIFF per channel):

```
<RUN_ID>_processed/stitched/
├─ cyc_1_cy5.tif
├─ cyc_1_TxRed.tif
├─ cyc_1_cy3.tif
├─ cyc_1_FAM.tif
└─ cyc_1_DAPI.tif
```

## 1. Probe Design (upstream)

Optional and not always necessary — you can design probes manually or contact us for help. To design probes in bulk, see [probe_designer](https://github.com/tangmc0210/probe_designer).

## 2. Image Processing (upstream — `spatial_img_core`)

PRISM starts from **stitched** images. The full acquisition chain — focal stacking, illumination correction (BaSiCPy / legacy CIDRE), per-cycle / per-channel registration, pcorr_bigstitcher / MIST stitching, and optional 3D AIRLOCALIZE — lives in the companion package `spatial_img_core` (**not yet public** — request access at **huanglab111@gmail.com**).

PRISM expects one stitched TIFF per channel under `<RUN_ID>_processed/stitched/`, e.g. `cyc_1_cy5.tif`, `cyc_1_TxRed.tif`, `cyc_1_cy3.tif`, `cyc_1_FAM.tif`, `cyc_1_DAPI.tif`.

## 3. Spot Detection / Readout

`scripts/readout.py` detects RNA spots on each stitched channel and reads out per-channel intensities. It is configured by `configs/readout.yaml` and takes the `RUN_ID` on the command line:

```bash
python scripts/readout.py <RUN_ID>
```

Edit `configs/readout.yaml` first:
- `base_dir`: the root that contains `<RUN_ID>_processed/`
- `channel_files`: the stitched TIFF filenames under `stitched/`
- `detection_method`: `spotiflow` (default, deep-learning), or the traditional `gaussian_tophat` / `tophat`
- `detection_snr`, `tophat_radius`, `search_radius`, `dedup_threshold`, `block_size`, `block_overlap`, `n_workers`

See the [Configuration Guide](configuration.md) for the full parameter reference.

**Outputs** (under `<RUN_ID>_processed/readout/`):
- `position.csv`: `index`, `Y`, `X`
- `intensity.csv`: `index` + one raw-intensity column per channel (original channel names)
- `readout_params.yaml`, `readout.log`

> Readout does **not** apply scaling, renaming, or crosstalk correction — those are handled in the gene-calling step.

### 3D Spot Detection

For confocal / light-sheet 3D stacks, we recommend [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE) for 3D spot extraction (it has a well-designed UI for parameter tuning). The Python wrapper and MATLAB tree now live in the companion `spatial_img_core` package — invoke them from there. Inputs come from `<RUN_ID>_processed/stitched/`, outputs go to `<RUN_ID>_processed/readout/`. Decode and call genes with `notebooks/readout_gene_calling_3D.ipynb`.

## 4. Gene Calling

`scripts/gene_calling.py` classifies each spot's intensity vector into a gene (or background). Set `BASE_DIR`, `RUN_ID`, and the config path at the top of the script, then run:

```bash
python scripts/gene_calling.py
```

The classification method is selected by `classification.method` in the config:
- `gmm` / `codebook_gmm` — Gaussian Mixture Model decoding (**default**)
- `postcode` — probabilistic decoding (**experimental**, opt-in; requires the vendored PoSTcode package, see the [Installation Guide](installation.md))

**Outputs** (under `<RUN_ID>_processed/readout/`):
- `mapping.csv`: per-spot gene assignment with top-1 / top-2 labels and probabilities
- `intensity_corrected.csv`: intensities after the 4×4 unmixing matrix (crosstalk / scaling / FRET), with fluorophore names

Channel correction (crosstalk, scaling, FRET) is applied here via the unmixing matrix in the config — use `scripts/calibrate_channels.py` to estimate the matrix from calibration experiments.

### Interactive gene calling

Because the color-space distribution varies between tissue types and cameras, gene calling often benefits from interactive inspection. Use the notebooks in `notebooks/`:
- `gene_calling_GMM.ipynb` / `readout_gene_calling_2D.ipynb` — GMM workflow (2D)
- `gene_calling_manual_2D.ipynb` — set thresholds per gene manually
- `gene_calling_mask_selection.ipynb` — mask-based selection
- `readout_gene_calling_3D.ipynb` / `gene_calling_manual_3D.ipynb` — 3D

## 5. Cell Segmentation

`scripts/segment_dapi.py` segments nuclei from the DAPI channel:

```bash
python scripts/segment_dapi.py <RUN_ID>
```

Options: `--base-dir`, `--nucleus-image` (default `cyc_1_DAPI.tif`), `--method` (`watershed` / `stardist` / `auto`), `--dimension` (`2d` / `3d` / `auto`).

**Outputs** (under `<RUN_ID>_processed/segmented/`):
- `dapi_centroids.csv`: nucleus centroids (`Y`, `X`; `Z` for 3D)
- a label image `*_labels.tif`

> 3D segmentation uses a trained StarDist network and needs the StarDist environment — see [StarDist](https://github.com/stardist/stardist) and the [Installation Guide](installation.md).

## 6. Cell-by-Gene Matrix & Cell Typing

The expression matrix assigns each RNA spot (`mapping.csv`) to its nearest nucleus (`dapi_centroids.csv`). This step, together with cell typing and downstream spatial analysis, is done in `notebooks/cell_typing_and_analysis.ipynb` (and `notebooks/co-expression_analysis.ipynb` for co-expression). If you have a better assignment strategy for your data, you can substitute it here.

## Additional Resources

- For probe design: [probe_designer](https://github.com/tangmc0210/probe_designer)
- For 3D segmentation: [StarDist](https://github.com/stardist/stardist)
- For 3D spot detection: [AIRLOCALIZE](https://github.com/timotheelionnet/AIRLOCALIZE) (invoked through `spatial_img_core`)

For questions or support, contact us at: **huanglab111@gmail.com**
