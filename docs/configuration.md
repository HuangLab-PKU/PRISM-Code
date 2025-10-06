# Configuration Guide

## Overview

The PRISM project uses YAML format configuration files to manage processing parameters across different pipeline components. This guide covers configuration for all major PRISM modules.

## Configuration File Structure

```
configs/
├── default_multi_channel_readout.yaml    # Multi-channel readout configuration
├── image_processing.yaml                # Image processing configuration (planned)
├── gene_calling.yaml                    # Gene calling configuration (planned)
├── cell_segmentation.yaml               # Cell segmentation configuration (planned)
└── analysis.yaml                        # Analysis configuration (planned)
```

## 1. Multi-Channel Readout Configuration

### Configuration Files
- `configs/default_multi_channel_readout.yaml`: Default configuration file

### Base Settings (base)

```yaml
base:
  channels: ['cy5', 'TxRed', 'cy3', 'FAM']  # Channel list
  base_dir: 'G:/spatial_data/processed'     # Data root directory
  prism_panel: 'PRISM30'                    # PRISM panel type
  max_memory: 32                            # Maximum memory usage (GB)
```

**Parameter Description**:
- `channels`: List of fluorescent channels used in the experiment
- `base_dir`: Root directory path for processed data
- `prism_panel`: PRISM panel type, supports PRISM30, PRISM31, PRISM45, PRISM46, PRISM63, PRISM64
- `max_memory`: Maximum memory used during processing, affects block size

### Image Processing Parameters (image_processing)

```yaml
image_processing:
  tophat_kernel_size: 7                    # Tophat kernel size
  tophat_break: 100                        # Tophat threshold
  local_max_abs_thre_ch:                   # Local maximum threshold for each channel
    cy5: 200
    TxRed: 200
    FAM: 200
    cy3: 200
  intensity_thre: null                     # Intensity threshold
  cal_snr: false                           # Whether to calculate SNR
```

**Parameter Description**:
- `tophat_kernel_size`: Kernel size for morphological operations, used for background removal
- `tophat_break`: Threshold after Tophat operation, pixels below this value are set to 0
- `local_max_abs_thre_ch`: Local maximum detection threshold for each channel
- `intensity_thre`: Intensity threshold for further spot filtering
- `cal_snr`: Whether to calculate signal-to-noise ratio

### Threshold Parameters (thresholds)

```yaml
thresholds:
  sum_threshold: 800                       # Sum intensity threshold
  g_abs_threshold: 1000                    # G channel absolute threshold
  g_threshold: 3                           # G channel relative threshold
  g_maxvalue: 5                            # G channel maximum value
```

**Parameter Description**:
- `sum_threshold`: Minimum sum of all channel intensities for spot filtering
- `g_abs_threshold`: Absolute threshold for G channel intensity
- `g_threshold`: Relative threshold for G channel (multiple of background)
- `g_maxvalue`: Maximum allowed value for G channel

### Processing Settings (batch)

```yaml
batch:
  overlap: 500                             # Overlap between processing blocks (pixels)
  max_volume_factor: 8                     # Factor for calculating max_volume
```

**Parameter Description**:
- `overlap`: Overlap size between adjacent processing blocks to avoid edge effects
- `max_volume_factor`: Factor used to calculate maximum processing volume based on available memory

### Signal Processing Parameters (signal_processing)

```yaml
signal_processing:
  snr: 8                                   # Signal-to-noise ratio threshold
  neighborhood_size: 10                    # Neighborhood size for SNR calculation
  kernel_size: 5                           # Kernel size for signal processing
  min_distance: 2                          # Minimum distance between spots
```

**Parameter Description**:
- `snr`: Signal-to-noise ratio threshold for spot filtering
- `neighborhood_size`: Size of neighborhood for SNR calculation
- `kernel_size`: Kernel size for signal processing operations
- `min_distance`: Minimum distance between detected spots

### Channel Mapping (channel_mapping)

```yaml
channel_mapping:
  cy5: 'R'                                 # Map cy5 to R channel
  TxRed: 'Ye'                              # Map TxRed to Ye channel
  cy3: 'G'                                 # Map cy3 to G channel
  FAM: 'B'                                 # Map FAM to B channel
```

**Parameter Description**:
- Maps original channel names to standardized output channel names
- Used for consistent naming across different experiments

### Channel Transformation Matrix (channel_transformation_matrix)

```yaml
channel_transformation_matrix:
  R: [1.0, 0.0, 0.0, 0.0]                 # R_corrected = R_raw * 1.0
  Ye: [0.0, 1.0, 0.0, 0.0]                # Ye_corrected = Ye_raw * 1.0
  G: [0.0, 0.0, 2.5, 0.0]                 # G_corrected = G_raw * 2.5
  B: [0.0, 0.0, -0.25, 0.75]              # B_corrected = G_raw * (-0.25) + B_raw * 0.75
```

**Parameter Description**:
- Transformation matrix for scaling factors and crosstalk elimination
- Format: [R_corrected, Ye_corrected, G_corrected, B_corrected] = matrix × [R_raw, Ye_raw, G_raw, B_raw]
- Matrix rows: [R, Ye, G, B] correspond to output channels
- Matrix columns: [R, Ye, G, B] correspond to input channels


## 2. Image Processing Configuration

*This section will be implemented in future versions.*

### Planned Configuration Files
- `configs/image_processing.yaml`: Image processing parameters
- `configs/stitching.yaml`: Image stitching parameters
- `configs/registration.yaml`: Image registration parameters

### Planned Parameters
- Image preprocessing settings
- Stitching algorithm parameters
- Registration method selection
- Quality control thresholds

## 3. Gene Calling Configuration

*This section will be implemented in future versions.*

### Planned Configuration Files
- `configs/gene_calling.yaml`: Gene calling parameters
- `configs/gmm.yaml`: Gaussian Mixture Model parameters
- `configs/manual_thresholds.yaml`: Manual threshold settings

### Planned Parameters
- GMM clustering parameters
- Manual threshold settings
- Quality filtering criteria
- Barcode mapping rules

## 4. Cell Segmentation Configuration

*This section will be implemented in future versions.*

### Planned Configuration Files
- `configs/cell_segmentation.yaml`: Cell segmentation parameters
- `configs/stardist.yaml`: StarDist model parameters
- `configs/watershed.yaml`: Watershed algorithm parameters

### Planned Parameters
- Segmentation method selection
- StarDist model configuration
- Watershed parameters
- Post-processing settings

## 5. Analysis Configuration

*This section will be implemented in future versions.*

### Planned Configuration Files
- `configs/cell_typing.yaml`: Cell typing parameters
- `configs/spatial_analysis.yaml`: Spatial analysis parameters
- `configs/visualization.yaml`: Visualization settings

### Planned Parameters
- Cell typing algorithms
- Spatial analysis methods
- Visualization preferences
- Output format settings

## Configuration Usage

### Loading Configuration

```python
from spot_detection import load_config

# Load default configuration
config = load_config()

# Load custom configuration
config = load_config('path/to/custom_config.yaml')
```

### Merging Configurations

```python
from spot_detection import merge_configs

# Merge base and override configurations
merged_config = merge_configs(base_config, override_config)
```


## Best Practices

### 1. Configuration Management
- Use version control for configuration files
- Document parameter changes in changelog
- Test configuration changes on small datasets first

### 2. Parameter Tuning
- Start with default parameters
- Adjust parameters based on data characteristics
- Validate changes with known datasets

### 3. Performance Optimization
- Adjust `max_memory` based on available system memory
- Optimize `overlap` for your data size
- Use appropriate `kernel_size` for your image resolution

### 4. Quality Control
- Monitor processing logs for parameter effectiveness
- Use visualization tools to validate parameter settings
- Document successful parameter combinations

## Troubleshooting

### Common Issues

#### Memory Errors
- Reduce `max_memory` parameter
- Increase `overlap` to reduce block size
- Process smaller datasets

#### Poor Spot Detection
- Adjust `local_max_abs_thre_ch` thresholds
- Modify `tophat_kernel_size`
- Check `intensity_thre` settings

#### Channel Crosstalk
- Update `channel_transformation_matrix`
- Verify `channel_mapping` settings
- Check fluorophore spectra

### Configuration Validation

The system automatically validates configuration files and provides error messages for:
- Missing required parameters
- Invalid parameter values
- Inconsistent channel mappings
- Malformed transformation matrices

## Future Development

### Planned Enhancements
1. **GUI Configuration Editor**: Visual interface for parameter adjustment
2. **Parameter Optimization**: Automated parameter tuning based on data characteristics
3. **Configuration Templates**: Pre-configured settings for common experimental setups
4. **Real-time Validation**: Live parameter validation during editing
5. **Configuration Sharing**: Export/import configuration profiles

### Contributing
When adding new configuration parameters:
1. Update this documentation
2. Add parameter validation
3. Include example configurations
4. Update changelog with new parameters