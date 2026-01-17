# Data Architecture

## Overview

This document describes the data structure and organization requirements for PRISM processing. Understanding the data architecture is crucial for successful analysis.

## Raw Data Structure

### Directory Organization

Raw data should be organized in the following structure:

```shell
Raw data root
├─RUN_ID1
│  └─cyc1
│     ├─C001-T0001-cy3-Z000.tif
│     ├─C001-T0001-cy3-Z001.tif
│     ├─...
│     ├─C001-T0004-FAM-Z006.tif
│     ├─...
│     └─C001-T0108-TxRed-Z008.tif
├─RUN_ID2
├─...
└─RUN_IDN
```

### File Naming Convention

The raw image files follow this naming pattern:
```
C{channel}-T{tile}-{fluorophore}-Z{z_slice}.tif
```

Where:
- `channel`: Channel number (e.g., C001, C002)
- `tile`: Tile number (e.g., T0001, T0002)
- `fluorophore`: Fluorophore name (cy3, cy5, FAM, TxRed)
- `z_slice`: Z-slice number (e.g., Z000, Z001)

### Example File Names
- `C001-T0001-cy3-Z000.tif`: Channel 1, Tile 1, cy3 fluorophore, Z-slice 0
- `C001-T0004-FAM-Z006.tif`: Channel 1, Tile 4, FAM fluorophore, Z-slice 6
- `C001-T0108-TxRed-Z008.tif`: Channel 1, Tile 108, TxRed fluorophore, Z-slice 8

## Processed Data Structure

### Directory Organization

Processed data is automatically organized in the following structure:

```shell
Output root
├─RUN_ID1_processed        # auto-created
│  ├─focal_stacked         # auto-created
│  ├─background_corrected  # auto-created, deleted after image processing
│  ├─resized               # auto-created, deleted after image processing
│  ├─registered            # auto-created
│  ├─stitched              # auto-created
│  ├─segmented             # auto-created
│  └─readout               # auto-created
├─RUN_ID2_processed        # auto-created
├─...
└─RUN_IDN_processed        # auto-created
```

### Directory Descriptions

#### Image Processing Directories
- `focal_stacked/`: Results from scan_fstack.py
- `background_corrected/`: Background correction results (temporary)
- `resized/`: Resized images (temporary)
- `registered/`: Registered images
- `stitched/`: Final stitched images

#### Analysis Directories
- `readout/`: Spot detection and intensity measurement results
- `segmented/`: Cell segmentation results
- `visualization/`: Output figures and visualizations

### File Contents

#### Stitched Directory
Contains final stitched images for each channel:
- `cyc_1_cy5.tif`
- `cyc_1_TxRed.tif`
- `cyc_1_cy3.tif`
- `cyc_1_FAM.tif`

#### Readout Directory
Contains spot detection results:
- `intensity_deduplicated.csv`: Final spot intensities
- `tmp/intensity_raw.csv`: Raw intensity measurements
- `tmp/tophat_mean.yaml`: Tophat statistics

#### Segmented Directory
Contains cell segmentation results:
- `dapi_centroids.csv`: Cell nucleus centroids
- `expression_matrix.csv`: Cell-by-gene expression matrix

## Configuration Variables

### Directory Path Variables

The following variables are used in the code to define directory paths:

```python
# In image processing
dest_dir = BASE_DIR / f'{RUN_ID}_processed' # processed data
aif_dir = dest_dir / 'focal_stacked'        # scan_fstack.py
sdc_dir = dest_dir / 'background_corrected' # image_process_after_stack.py
rsz_dir = dest_dir / 'resized'              # image_process_after_stack.py
rgs_dir = dest_dir / 'registered'           # image_process_after_stack.py
stc_dir = dest_dir / 'stitched'             # image_process_after_stack.py

# In subsequent analysis
src_dir = BASE_DIR / f'{RUN_ID}_processed'  # processed data
stc_dir = src_dir / 'stitched'              # image_process_after_stack.py
read_dir = src_dir / 'readout'              # multi_channel_readout.py
seg_dir = src_dir / 'segmented'             # segment2D.py or segment3D.py or expression_matrix.py
visual_dir = src_dir / 'visualization'      # folder for figures...
```

## Data Requirements

### File Formats
- **Input**: TIFF files (.tif)
- **Output**: CSV files (.csv) for data, YAML files (.yaml) for configuration

### Image Specifications
- **Bit Depth**: 16-bit recommended
- **Channels**: Multi-channel TIFF files
- **Compression**: Uncompressed or LZW compression

### Storage Requirements
- **Temporary Space**: At least 2x the size of raw data
- **Final Storage**: Approximately 10-20% of raw data size
- **Processing Space**: Additional 50% for intermediate files

## Data Flow

### Processing Pipeline
1. **Raw Images** → `focal_stacked/`
2. **Focal Stacked** → `background_corrected/` → `resized/` → `registered/` → `stitched/`
3. **Stitched Images** → `readout/`
4. **Readout Results** → `segmented/`

### File Dependencies
- Spot detection requires stitched images
- Gene calling requires readout results
- Cell segmentation requires both readout and DAPI images
- Expression matrix requires both gene calling and cell segmentation results

## Best Practices

### Data Organization
1. Use consistent RUN_ID naming
2. Keep raw and processed data separate
3. Maintain directory structure integrity
4. Regular cleanup of temporary files

### Performance Optimization
1. Use SSD storage for temporary files
2. Ensure sufficient disk space
3. Monitor disk usage during processing
4. Clean up intermediate files after completion

### Backup Strategy
1. Backup raw data before processing
2. Keep processed results in separate location
3. Version control configuration files
4. Document processing parameters

## Troubleshooting

### Common Issues
1. **Missing Files**: Check file naming convention
2. **Path Errors**: Verify directory structure
3. **Permission Issues**: Check file permissions
4. **Disk Space**: Monitor available storage

### Validation
1. Verify file counts in each directory
2. Check file sizes and formats
3. Validate directory structure
4. Test file accessibility

