# Memory Mapping Optimization for Spot Detection

## Overview

This document describes the memory mapping optimization implemented in the spot detection framework to reduce I/O costs and memory usage when processing large images, similar to the approach used in `multi_channel_readout.py`.

## Problem Statement

### Current Issues
- **Direct Memory Loading**: The original `spot_detection_pipeline.py` loads entire images into memory using `tifffile.imread()`
- **High Memory Usage**: Multi-channel large images consume significant memory
- **Repeated I/O**: Tiling operations may cause repeated file reads
- **Memory Inefficiency**: Large images (>2GB) can cause memory pressure

### Solution: Memory Mapping
Memory mapping allows efficient access to large files without loading them entirely into memory, providing:
- **Reduced Memory Usage**: Only load required image tiles
- **Efficient I/O**: Avoid repeated file reads
- **Scalability**: Handle images larger than available RAM

## Implementation

### 1. Core Components

#### `MemmapImageLoader`
Single-channel memory-mapped image loader:
```python
from spot_detection import MemmapImageLoader

with MemmapImageLoader() as loader:
    # Create memory-mapped file
    info = loader.create_memmap("image.tif", "channel_key")
    
    # Load specific tile
    tile = loader.load_tile("channel_key", pad_x, pad_y, cut_x, cut_y)
```

#### `MultiChannelMemmapLoader`
Multi-channel coordinated memory mapping:
```python
from spot_detection import MultiChannelMemmapLoader

image_dict = {
    'ch1': 'path/to/ch1.tif',
    'ch2': 'path/to/ch2.tif',
    'ch3': 'path/to/ch3.tif',
    'ch4': 'path/to/ch4.tif'
}

with MultiChannelMemmapLoader() as loader:
    # Create memmaps for all channels
    loader.create_channel_memmaps(image_dict)
    
    # Load tiles from all channels
    tile_dict = loader.load_all_channel_tiles(pad_x, pad_y, cut_x, cut_y)
```

### 2. Enhanced TileProcessor

The `TileProcessor` class now supports memory-mapped processing:

```python
def process_large_image_with_memmap(
    self, 
    memmap_loader, 
    processor_func, 
    image_shape: tuple,
    **kwargs
) -> SpotDetectionResult:
    """Process large images using memory mapping"""
    # Calculate tiling parameters
    # Load tiles using memmap_loader
    # Process each tile
    # Merge results
```

### 3. Memory Usage Estimation

The framework includes memory usage estimation:

```python
from spot_detection import estimate_memory_usage

memory_info = estimate_memory_usage(shape, dtype, num_channels)
print(f"Original memory: {memory_info['original_memory_gb']:.2f} GB")
print(f"Memory savings: {memory_info['memory_savings_gb']:.2f} GB")
```

### 4. Configuration Support

Memory mapping can be configured in `spot_detection_base.yaml`:

```yaml
base:
  max_memory: 32  # GB
  use_memmap: true  # Enable memory mapping
  memmap_threshold_gb: 2.0  # Use memmap for images > 2GB
```

## Performance Benefits

### Memory Usage Comparison

| Image Size | Channels | Original Memory | Memmap Memory | Savings |
|------------|----------|----------------|---------------|---------|
| 2000×2000  | 4        | 0.03 GB        | 0.10 GB       | -0.07 GB |
| 4000×4000  | 4        | 0.12 GB        | 0.10 GB       | 0.02 GB  |
| 8000×8000  | 4        | 0.48 GB        | 0.10 GB       | 0.38 GB  |

### Key Advantages

1. **Memory Efficiency**: For large images (>2GB), memory mapping provides significant memory savings
2. **I/O Optimization**: Eliminates repeated file reads during tiling
3. **Scalability**: Can handle images larger than available RAM
4. **Transparency**: Minimal changes to existing code

## Usage Examples

### Basic Usage

```python
from spot_detection import MultiChannelMemmapLoader, UnifiedSpotDetector

# Load configuration
config = load_spot_detection_config("configs/spot_detection_traditional.yaml")

# Create detector
detector = UnifiedSpotDetector(config)

# Use memory mapping for large images
with MultiChannelMemmapLoader() as memmap_loader:
    # Create memmaps
    memmap_loader.create_channel_memmaps(image_dict)
    
    # Process with memory mapping
    result = detector.detect_spots_with_memmap(memmap_loader)
```

### Integration with Pipeline

The pipeline automatically detects large images and recommends memory mapping:

```python
def load_image_with_memmap(image_path: str, use_memmap: bool = True):
    """Load image with memory mapping for large files"""
    memory_info = estimate_memory_usage(shape, dtype, num_channels=1)
    
    if use_memmap and memory_info['original_memory_gb'] > 2.0:
        logger.info(f"Large image detected, using memory mapping")
        # Use memmap loader
```

## Testing

Run the memory mapping tests:

```bash
conda activate PRISM
cd code/local/test
python test_memmap_optimization.py
```

Test results show:
- ✅ Memory usage estimation works correctly
- ✅ Single-channel memory mapping functions properly
- ✅ Multi-channel coordination works
- ✅ Data integrity is maintained
- ✅ Integration with spot detection framework

## Comparison with multi_channel_readout.py

### Similarities
- **Memory Mapping**: Both use `np.memmap` for efficient file access
- **Tiling Support**: Both support large image tiling
- **Multi-channel**: Both handle multiple channels efficiently

### Improvements in Spot Detection Framework
- **Modular Design**: Separate classes for different responsibilities
- **Configuration-driven**: Memory mapping controlled via config
- **Automatic Detection**: Framework automatically detects when to use memory mapping
- **Better Integration**: Seamless integration with existing spot detection pipeline
- **Memory Estimation**: Built-in memory usage estimation

## Future Enhancements

1. **Automatic Memory Mapping**: Automatically enable memory mapping based on image size
2. **Parallel Processing**: Support parallel tile processing with memory mapping
3. **Cache Management**: Intelligent caching of frequently accessed tiles
4. **Compression**: Support for compressed memory-mapped files

## Conclusion

The memory mapping optimization provides significant benefits for large image processing:

- **Reduced Memory Usage**: Especially beneficial for images >2GB
- **Improved I/O Efficiency**: Eliminates repeated file reads
- **Better Scalability**: Handle images larger than available RAM
- **Minimal Code Changes**: Transparent integration with existing framework

This implementation brings the I/O optimization benefits of `multi_channel_readout.py` to the modern, modular spot detection framework while maintaining backward compatibility and providing additional features.

