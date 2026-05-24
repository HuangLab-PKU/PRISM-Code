# Quick Start

## Complete Workflow Overview

PRISM is the **post-stitching** half of the pipeline: `readout` → `gene_calling` → `cell_segmentation` → `analysis`. Upstream (probe design → experiment → raw images → stitched images) lives in the sibling [`probe_designer`](https://github.com/tangmc0210/probe_designer) and `spatial_img_core` (sibling repo) packages.

```mermaid
graph TD
    A["Probe Design (probe_designer)"] --> B["Experiment"];
    B --> C["Raw Images (2D / 3D)"];

    subgraph "Upstream: spatial_img_core"
        C --> D["Stitched Images"];
    end

    D --> E("Spot Detection / Readout");
    E --> F("Gene Calling");

    subgraph "PRISM (this repo)"
        F --> G["Cell Segmentation"];
        G --> H["Analysis"];
    end
```

## Quick Start Options

### Option A: Start from Stitched Images (Recommended for Beginners)

**Prerequisites:**
- Download sample data from Zenodo: [MouseEmbryo Data](https://zenodo.org/records/13219763)
- Organize data according to [Data Architecture](data-architecture.md)

**Workflow Steps:**

1. **Signal Detection**
   ```bash
   python scripts/readout.py
   ```

2. **Gene Calling**
   - Use the provided notebook: `PRISM_gene_calling_EMBRYO_30.ipynb`
   - Location: `20221219_PRISM_E13.5_2_3_Three_processed/readout/PRISM_gene_calling_EMBRYO_30.ipynb`

3. **Cell Segmentation**
   ```bash
   python scripts/segment_dapi.py
   ```

**Expected Outputs:**
- `mapped_genes.csv`: RNA spots with spatial coordinates and gene assignments
- `dapi_centroids.csv`: Cell nucleus centroids from DAPI segmentation
- `expression_matrix.csv`: Cell-by-gene expression count matrix

### Option B: Start from Raw Unstitched Images (Full Pipeline)

PRISM no longer ships an in-tree image-processing layer. For raw → stitched output, use the sibling `spatial_img_core` (sibling repo) package (focal stacking, illumination correction, registration, pcorr_bigstitcher / MIST stitching, AIRLOCALIZE). Once stitched images land under `<run_id>_processed/stitched/`, continue with **Option A** above.

## Data Acquisition

Stitched raw images are provided on zenodo.org, download based on your needs:

1. [MouseEmbryo](https://zenodo.org/records/13219763)
2. [HCC](https://zenodo.org/records/13208941)
3. [MouseBrain3D](https://zenodo.org/records/12673246)
4. [Cell typing and Analysis](https://zenodo.org/records/12755414)

We also provide **HCC2D** unstitched raw images on [PKU NetDisk](https://disk.pku.edu.cn/link/AA83FADBB90EB14BAE8E9DE5889E94AFF9).

**For more raw data, contact us: huanglab111@gmail.com.**

## Refactored Multi-Channel Readout (Optional)

For users who want to use the refactored, more modular version of the multi-channel readout script:

### Basic Usage

```python
from scripts.multi_channel_readout_refactored import MultiChannelProcessor

# Process single dataset with default configuration
processor = MultiChannelProcessor(run_id='20250717_FFPE_OSCC')
intensity = processor.process_single_run()
```

### Command Line Usage

```bash
# Single file processing
python scripts/multi_channel_readout_refactored.py
```

## Next Steps

- View [Detailed Usage Guide](detailed-usage.md) for comprehensive workflow instructions
- Understand [Data Architecture](data-architecture.md) requirements
- Refer to [Installation Guide](installation.md) for environment setup
- Check [Tutorial](tutorial.md) for step-by-step operations
- Explore [Configuration Guide](configuration.md) for parameter customization