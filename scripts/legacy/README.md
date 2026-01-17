# PRISM Scripts Documentation

This directory contains convenient script interfaces for running PRISM analysis workflows.

## Available Scripts

### Cell Segmentation

#### `segment_cells.py`

Simple interface for cell segmentation using the unified segmentation module.

**Usage:**
```bash
python segment_cells.py
```

**Configuration:**
Modify the variables in the main function to match your data:
```python
BASE_DIR = 'path_to_processed_dataset'
RUN_ID = 'example_data'
CELL_IMAGE_NAME = 'cyc_1_DAPI.tif'
METHOD = 'auto'      # 'watershed', 'stardist', 'auto'
DIMENSION = 'auto'   # '2d', '3d', 'auto'
```

---

## Cell Segmentation Module Documentation

### Overview

The unified cell segmentation module integrates three original segmentation scripts (`segment2D.py`, `segment2D_stardist.py`, `segment3D.py`) into a single functional module that supports multiple segmentation methods and automatic dimension detection.

### Main Features

#### 1. Multiple Segmentation Methods
- **Watershed**: Traditional watershed algorithm-based segmentation (2D only)
- **StarDist**: Deep learning-based star-convex segmentation (2D and 3D support)

#### 2. Automatic Detection
- **Dimension Auto-detection**: Automatically identifies whether input image is 2D or 3D
- **Method Auto-selection**: Automatically selects the best segmentation method based on availability

#### 3. Flexible Configuration System
- **SegmentationConfig class**: Centralized parameter management
- **Categorized parameters**: Watershed, StarDist 2D/3D, and general parameters managed separately

#### 4. Large Image Support
- **Block processing**: Supports block processing for very large images
- **Overlap stitching**: Automatically handles overlapping regions between blocks
- **Deduplication**: Automatically removes duplicate detected cells

### Usage Methods

#### Basic Usage

```python
from cell_segmentation.unified_segmentation import UnifiedSegmentation
from cell_segmentation.segmentation_config import SegmentationConfig

# Create configuration
config = SegmentationConfig()
config.general_params['base_dir'] = Path('/path/to/your/data')
config.general_params['run_id'] = 'experiment_name'

# Create segmenter
segmenter = UnifiedSegmentation(config)

# Execute segmentation
results = segmenter.segment(
    image_path=Path('image.tif'),
    method='auto',      # 'watershed', 'stardist', 'auto'
    dimension='auto',   # '2d', '3d', 'auto'
    output_dir=Path('output')
)
```

#### Method Parameters

##### segment() method
- `image_path`: Input image path
- `method`: Segmentation method
  - `'watershed'`: Watershed algorithm (2D only)
  - `'stardist'`: StarDist algorithm (2D/3D)
  - `'auto'`: Auto-select (prioritizes StarDist)
- `dimension`: Image dimension
  - `'2d'`: Force 2D processing
  - `'3d'`: Force 3D processing
  - `'auto'`: Auto-detect
- `output_dir`: Output directory (optional)

##### Return Results
```python
results = {
    'method': 'segmentation_method_used',
    'dimension': 'image_dimension',
    'coordinates': 'cell_center_coordinates_array',
    'labels': 'segmentation_label_image',
    'dataframe': 'results_dataframe'
}
```

### Configuration Parameters

#### Watershed Parameters
```python
config.watershed_params = {
    'min_region': 400,          # Minimum region size
    'min_cell_size': 400,       # Minimum cell size
    'block_size': 20000,        # Block size
    'offset_value': 255,        # Threshold offset
    'threshold_block_size': 251, # Adaptive threshold block size
    'min_distance': 7,          # Peak minimum distance
    'dilate_kernel': 3,         # Dilation kernel size
    'erode_kernel': 2           # Erosion kernel size
}
```

#### StarDist 2D Parameters
```python
config.stardist_2d_params = {
    'model_name': '2D_versatile_fluo',  # Pre-trained model name
    'block_size': [5000, 5000],         # Block size
    'min_overlap': [448, 448],          # Minimum overlap
    'context': [94, 94],                # Context size
    'min_cell_size': 400,               # Minimum cell size
    'predict_kwargs': {'verbose': 0}     # Prediction parameters
}
```

#### StarDist 3D Parameters
```python
config.stardist_3d_params = {
    'model_path': './models',           # Model path
    'model_name': 'stardist',          # Model name
    'max_size': 512 * 512 * 150,      # Maximum memory usage
    'min_overlap': [30, 90, 90],       # Minimum overlap
    'context': [20, 40, 40],           # Context size
    'predict_kwargs': {'verbose': 0}    # Prediction parameters
}
```

#### General Parameters
```python
config.general_params = {
    'base_dir': Path('path_to_processed_dataset'),  # Base directory
    'run_id': 'example_data',                       # Experiment ID
    'cell_image_name': 'cyc_1_DAPI.tif',           # Cell image name
    'duplicate_distance': 5                         # Deduplication distance threshold
}
```

### Output Files

#### Standard Output
- `{image_name}_{method}_{dimension}_labels.tif`: Segmentation label image
- `{image_name}_{method}_{dimension}_centroids.csv`: Cell center coordinates

#### Compatibility Output (StarDist 2D)
- `dapi_centroids.csv`: Cell center coordinates (compatible format)
- `dapi_predict.csv`: Complete results with probabilities

#### Block Processing Output (Watershed)
- `centroids_y_{y}_x_{x}.csv`: Coordinate files for each block
- `segmented_y_{y}_x_{x}.tif`: Segmentation results for each block
- `dapi_centroids.csv`: Final merged coordinates

### Dependencies

#### Required Dependencies
```
numpy
pandas
scipy
scikit-image
opencv-python
tifffile
tqdm
pathlib
```

#### Optional Dependencies (StarDist functionality)
```
stardist
csbdeep
tensorflow  # or tensorflow-gpu
```

### Usage Examples

#### 1. Automatic Segmentation
```python
# Simplest usage - fully automatic
config = SegmentationConfig()
segmenter = UnifiedSegmentation(config)

results = segmenter.segment(
    image_path=Path('cell_image.tif'),
    method='auto',
    dimension='auto',
    output_dir=Path('results')
)

print(f"Detected {len(results['dataframe'])} cells")
```

#### 2. Specify Method and Parameters
```python
# Use specific method with custom parameters
config = SegmentationConfig()
config.watershed_params['min_cell_size'] = 200
config.stardist_2d_params['min_cell_size'] = 300

segmenter = UnifiedSegmentation(config)

results = segmenter.segment(
    image_path=Path('cell_image.tif'),
    method='stardist',
    dimension='2d',
    output_dir=Path('results')
)
```

#### 3. Large Image Block Processing
```python
# For Watershed method only
config = SegmentationConfig()
config.watershed_params['block_size'] = 10000

segmenter = UnifiedSegmentation(config)

# Block processing
img = imread('large_image.tif')
temp_dir = Path('temp')
segmenter.block_segment_watershed(img, temp_dir)

# Combine results
output_dir = Path('final_results')
segmenter.combine_centroids(temp_dir, output_dir)
```

### Performance Optimization

#### 1. Memory Management
- Use block processing for large images
- Adjust `block_size` parameter appropriately
- Pay attention to `max_size` parameter for 3D image processing

#### 2. Speed Optimization
- GPU support: Install `tensorflow-gpu` for better StarDist performance
- Parallel processing: Can process multiple small images simultaneously
- Parameter tuning: Adjust parameters based on data characteristics

#### 3. Quality Control
- Check if `min_cell_size` parameter suits your data
- Verify auto-detected dimensions are correct
- For special data, manually specify method and dimension

### Troubleshooting

#### Common Errors and Solutions

1. **StarDist not available**
   ```
   Warning: StarDist not available. Only watershed method can be used.
   ```
   Solution: Install StarDist related dependencies

2. **Dimension detection error**
   ```
   ValueError: Unsupported image dimension
   ```
   Solution: Manually specify `dimension` parameter

3. **Method incompatibility**
   ```
   ValueError: Watershed method only supports 2D images
   ```
   Solution: Use StarDist method or convert 3D image to 2D

4. **Out of memory**
   ```
   OutOfMemoryError
   ```
   Solution: Use block processing or reduce `block_size`

---

## Other Workflows

### Image Processing Pipeline
*Documentation to be added*

### Gene Calling
*Documentation to be added*

### Spot Detection
*Documentation to be added*

### Data Analysis
*Documentation to be added*

---

## Contributing

Welcome to submit issue reports and feature requests. To contribute code, please:

1. Fork the project
2. Create a feature branch
3. Submit changes
4. Create a Pull Request

## License

This project uses the same license as the original PRISM project.