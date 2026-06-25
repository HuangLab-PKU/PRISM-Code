# Quick Start

PRISM is the **post-stitching** half of the pipeline: `readout` → `gene_calling` → `cell_segmentation` → `analysis`. For the pipeline overview and diagram, see the [main README](../README.md); for sample-data downloads, see [Data Sources](../README.md#data-sources).

Install the package first — see the [Installation Guide](installation.md). All commands below run from the repository root.

## Option A: Start from Stitched Images (Recommended for Beginners)

**Prerequisites:**
- Download sample data from Zenodo (e.g. [MouseEmbryo](https://zenodo.org/records/13219763)); see [Data Sources](../README.md#data-sources) for the full list.
- Organize data according to [Data Architecture](data-architecture.md): stitched per-channel TIFFs under `<RUN_ID>_processed/stitched/`.

**Workflow Steps:**

1. **Spot Detection / Readout** — edit `base_dir` (and, if needed, `channel_files`) in `configs/readout.yaml`, then run:
   ```bash
   python scripts/readout.py <RUN_ID>
   ```
   Outputs `readout/position.csv` and `readout/intensity.csv`.

2. **Gene Calling** — set `BASE_DIR` and `RUN_ID` at the top of `scripts/gene_calling.py` (or pass a config file), then run:
   ```bash
   python scripts/gene_calling.py
   ```
   Outputs `readout/mapping.csv` (gene assignments + confidence) and `readout/intensity_corrected.csv`. The default method is GMM / codebook-GMM.
   For an interactive walkthrough, use `notebooks/readout_gene_calling_2D.ipynb` or `notebooks/gene_calling_GMM.ipynb`.

3. **Cell Segmentation** — segment nuclei from the DAPI channel:
   ```bash
   python scripts/segment_dapi.py <RUN_ID>
   ```
   Outputs nucleus centroids (`segmented/dapi_centroids.csv`) and a label image.

4. **Cell-by-gene matrix & cell typing** — assign RNA spots (`mapping.csv`) to their nearest nucleus and build the expression matrix in `notebooks/cell_typing_and_analysis.ipynb`.

**Expected Outputs:**
- `readout/position.csv`, `readout/intensity.csv`: detected RNA spots (coordinates + per-channel intensities)
- `readout/mapping.csv`: spots with gene assignments and confidence
- `segmented/dapi_centroids.csv`: nucleus centroids
- cell-by-gene expression matrix: built in the cell-typing notebook

## Option B: Start from Raw Unstitched Images (Full Pipeline)

PRISM does not ship an in-tree image-processing layer. For raw → stitched output (focal stacking, illumination correction, registration, pcorr_bigstitcher / MIST stitching, AIRLOCALIZE), use the companion `spatial_img_core` package (**not yet public** — request access at **huanglab111@gmail.com**). Once stitched images land under `<RUN_ID>_processed/stitched/`, continue with **Option A** above.

## Next Steps

- View [Detailed Usage Guide](detailed-usage.md) for comprehensive workflow instructions
- Understand [Data Architecture](data-architecture.md) requirements
- Refer to [Installation Guide](installation.md) for environment setup
- Explore [Configuration Guide](configuration.md) for parameter customization
