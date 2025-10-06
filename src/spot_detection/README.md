# Spot Detection Framework

A maintainable and extensible spot detection framework supporting multiple detection methods and intensity extraction strategies.

## Key Features

- **Modular Design**: Supports traditional and deep learning methods
- **Extensibility**: Easy to add new background removal, coordinate detection, and intensity extraction methods
- **Image Sharing**: Smart caching of processed images to avoid redundant computation
- **Large Image Support**: Automatic tiling with dynamic memory-based thresholds
- **Multi-channel Support**: Unified channel system (ch1, ch2, ch3, ch4) with matrix-based corrections
- **Configuration-driven**: Modular configuration system with separate files for different components
- **Memory Management**: Intelligent memory usage based on available RAM and channel count
- **Coordinate Deduplication**: Automatic removal of duplicate coordinates across channels
- **Output Separation**: Separate raw and transformed intensity files for downstream analysis

## Quick Start

### Basic Usage

```python
from spot_detection import detect_spots
import numpy as np

# Create test image
image = np.random.rand(1000, 1000) * 1000

# Detect spots
result = detect_spots(image, method='traditional', intensity_method='gaussian')

print(f"Detected {len(result.coordinates)} spots")
print(f"Intensity range: {result.intensities['intensity'].min():.2f} - {result.intensities['intensity'].max():.2f}")
```

### Using Preset Configurations

```python
from spot_detection import create_preset_detector

# Create fast detector
detector = create_preset_detector('traditional_fast')
result = detector.detect_spots(image)

# Create accurate detector
detector = create_preset_detector('traditional_accurate')
result = detector.detect_spots(image)
```

### Multi-channel Processing

```python
# Multi-channel image dictionary using unified channel names
image_dict = {
    'ch1': cy5_image,      # 670nm - Cy5
    'ch2': txred_image,    # 615nm - Texas Red  
    'ch3': cy3_image,      # 550nm - Cy3
    'ch4': fam_image       # 520nm - FAM
}

# Detect multi-channel spots
result = detect_spots(image_dict, method='traditional', intensity_method='direct')

# Access results
print(f"Total spots: {len(result.coordinates)}")
print(f"Intensity columns: {list(result.intensities.keys())}")
```

## Documentation

For detailed information, see the comprehensive documentation:

- **[Installation Guide](docs/installation.md)** - Setup and installation instructions
- **[Detailed Usage](docs/detailed_usage.md)** - Comprehensive usage examples and API reference
- **[Configuration Guide](docs/configuration.md)** - Configuration file options and examples
- **[Detection Methods](docs/detection_methods.md)** - Available detection algorithms
- **[Intensity Extraction](docs/intensity_extraction.md)** - Intensity extraction strategies
- **[Advanced Usage](docs/advanced_usage.md)** - Custom detectors and framework extension
- **[Performance Guide](docs/performance.md)** - Optimization and performance tuning

## Command Line Usage

The main execution script is `spot_detection_pipeline.py` which provides a unified interface for both single and multi-channel spot detection:

```bash
# Multi-channel processing (recommended)
python scripts/spot_detection_pipeline.py \
    --input stitched \
    --run-id 20230523_HCC_PRISM_probe_refined_crop \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml

# With custom output directory
python scripts/spot_detection_pipeline.py \
    --input stitched \
    --run-id your_run_id \
    --data-dir /path/to/data \
    --output /path/to/output \
    --config code/configs/spot_detection.yaml

# Verbose output for debugging
python scripts/spot_detection_pipeline.py \
    --input stitched \
    --run-id your_run_id \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml \
    --verbose
```

### Command Line Arguments

- `--input, -i`: Input image directory (e.g., "stitched" subdirectory)
- `--run-id`: Run ID for processing (required)
- `--data-dir`: Base data directory (default: "G:/spatial_data")
- `--output, -o`: Output directory (default: {data_dir}/processed/{run_id}/readout)
- `--config`: Configuration file path (required)
- `--multi-channel`: Process as multi-channel (default: True)
- `--verbose, -v`: Verbose output for debugging

### Output Files

The pipeline generates the following output files in the `readout` directory:

- `intensity_raw.csv`: Raw intensity values (ch1, ch2, ch3, ch4)
- `intensity.csv`: Transformed intensity values (after matrix correction)
- `coordinates.csv`: Spot coordinates (Y, X)
- `metadata.yaml`: Processing metadata and parameters
- `channel_info.yaml`: Channel configuration and mapping information
- `full_config.yaml`: Complete merged configuration used
- `spot_detection.yaml`: Original configuration file copy

## Configuration System

The framework uses a modular configuration system with separate files for different components:

- `spot_detection_base.yaml` - Base configuration (channels, memory, tiling)
- `spot_detection_traditional.yaml` - Traditional detection parameters
- `spot_detection_deep_learning.yaml` - Deep learning detection parameters  
- `spot_detection_intensity.yaml` - Intensity extraction parameters
- `spot_detection.yaml` - Main configuration file that loads all modules

### Key Features

- **Unified Channel System**: Uses ch1, ch2, ch3, ch4 for consistent channel naming
- **Dynamic Memory Management**: Automatically calculates tiling thresholds based on available memory and channel count
- **Matrix-based Corrections**: Supports crosstalk correction and scaling transformations
- **Modular Design**: Easy to extend with new detection methods and intensity extractors

## Dependencies

- numpy
- pandas
- opencv-python
- scikit-image
- scipy
- tqdm
- pyyaml
- stardist (for deep learning methods)

## Backward Compatibility

The framework maintains backward compatibility with existing `multi_channel_readout` functions. Legacy functions are available in the `legacy` module:

```python
from spot_detection import tophat_spots, extract_coordinates

# Legacy functions still available
tophat_image = tophat_spots(image)
coordinates = extract_coordinates(tophat_image)
```

### Legacy Files

The following legacy files have been moved to `scripts/legacy/` for reference:

- `multi_channel_readout.py` - Original multi-channel processing script
- `multi_channel_readout_refactored.py` - Refactored version
- `multi_channel_readout_dp.py` - Data processing version

These files are preserved for backward compatibility and reference, but the new unified framework is recommended for all new projects.

## License

This project follows the project license.