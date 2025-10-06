# PRISM Scripts Documentation

This directory contains convenient script interfaces for running PRISM analysis workflows.

## Available Scripts

Scripts are organized in the order they are typically run in the PRISM workflow:

### 1. Image Processing

#### `image_scan_fstack.py`

Processes raw images captured in small fields and multiple channels to generate focal stacked images.

**Usage:**
```bash
python image_scan_fstack.py Raw_data_root
```

**Output:** Creates `focal_stacked` directory with processed images.

---

#### `image_process_pipeline.py`

Complete image processing pipeline including registration, background correction, and stitching.

**Usage:**
```bash
python image_process_pipeline.py
```

**Output:** Creates `stitched` directory with final multi-channel images.

---

#### `pipeline_3D.py`

3D reconstruction pipeline for z-stack images (optional, for 3D analysis).

**Usage:**
```bash
python pipeline_3D.py
```

**Output:** 3D reconstructed images for 3D spot detection.

---

### 2. Spot Detection

#### `spot_detection_pipeline.py` ⭐ **Recommended**

Unified spot detection pipeline for multi-channel image processing with traditional and deep learning methods.

**Quick Start:**
```bash
python spot_detection_pipeline.py \
    --input stitched \
    --run-id your_run_id \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml
```

**Key Features:**
- Multi-channel processing with unified channel system (ch1, ch2, ch3, ch4)
- Dynamic memory management and automatic tiling
- Matrix-based crosstalk correction
- Coordinate deduplication across channels
- Modular configuration system

**Output:** Creates `readout` directory with intensity and coordinate files.

---

#### `train_spot_detector.py`

Train StarDist deep learning models for spot detection (optional, for improved accuracy).

**Usage:**
```bash
python train_spot_detector.py --use-gpu
```

**Output:** Trained models saved to `models/` directory.

---

### 3. Gene Calling

#### `gene_calling.py` ⭐ **Recommended**

Unified gene calling pipeline for signal point classification using GMM and other methods.

**Quick Start:**
```bash
python gene_calling.py \
    --data-dir /path/to/data \
    --run-id experiment_name \
    --config code/configs/gene_calling.yaml
```

**Key Features:**
- GMM-based signal point classification
- Modular configuration system
- Integration with spot detection output
- Manual threshold adjustment capabilities
- Quantitative evaluation tools

**Output:** Creates `classification` directory with prediction results and evaluation reports.

---

#### Legacy Gene Calling Scripts

Legacy gene calling scripts preserved in `legacy/` directory for backward compatibility:

- `legacy/gene_calling_1_preprocess.py` - Preprocessing step
- `legacy/gene_calling_2_GMM.py` - GMM-based classification
- `legacy/gene_calling_3_optim.py` - Optimization step
- `legacy/gene_calling_pipeline_GUI.py` - GUI-based pipeline

**Note:** These are maintained for backward compatibility. Use `gene_calling.py` for new projects.

---

### 4. Cell Segmentation

#### `segment_cell_2D.py`

2D cell segmentation using DAPI channel.

**Usage:**
```bash
python segment_cell_2D.py
```

**Output:** Cell centroids in `centroids_all.csv`.

---

#### `segment_cell_3D.py`

3D cell segmentation using StarDist (requires StarDist environment).

**Usage:**
```bash
python segment_cell_3D.py
```

**Output:** 3D cell segmentation results.

---

#### `segment_dapi.py`

DAPI-based cell segmentation.

**Usage:**
```bash
python segment_dapi.py
```

**Output:** DAPI-based cell centroids.

---

### 5. Legacy Scripts

#### Legacy Multi-Channel Readout

Legacy spot detection scripts preserved in `legacy/` directory for backward compatibility:

- `legacy/multi_channel_readout.py` - Original multi-channel processing script
- `legacy/multi_channel_readout_refactored.py` - Refactored version
- `legacy/multi_channel_readout_dp.py` - Data processing version

**Note:** These are maintained for backward compatibility. Use `spot_detection_pipeline.py` for new projects.

---

## Workflow Overview

The typical PRISM analysis workflow follows this sequence:

1. **Image Processing** → 2. **Spot Detection** → 3. **Gene Calling** → 4. **Cell Segmentation**

Each step builds upon the previous one, with outputs from one step serving as inputs for the next.

### Data Flow

```
Raw Images → Focal Stacking → Registration & Stitching → Spot Detection → Gene Calling → Cell Segmentation
     ↓              ↓                    ↓                    ↓              ↓              ↓
focal_stacked → background_corrected → stitched → readout → mapped_genes → segmented
```

---

## Detailed Documentation

### Spot Detection Pipeline

#### `spot_detection_pipeline.py`

Unified spot detection pipeline supporting both traditional and deep learning methods with multi-channel processing capabilities.

**Usage:**
```bash
# Multi-channel processing (recommended)
python spot_detection_pipeline.py \
    --input stitched \
    --run-id 20230523_HCC_PRISM_probe_refined_crop \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml

# With custom output directory
python spot_detection_pipeline.py \
    --input stitched \
    --run-id your_run_id \
    --data-dir /path/to/data \
    --output /path/to/output \
    --config code/configs/spot_detection.yaml

# Verbose output for debugging
python spot_detection_pipeline.py \
    --input stitched \
    --run-id your_run_id \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml \
    --verbose
```

**Command Line Arguments:**
- `--input, -i`: Input image directory (e.g., "stitched" subdirectory)
- `--run-id`: Run ID for processing (required)
- `--data-dir`: Base data directory (default: "G:/spatial_data")
- `--output, -o`: Output directory (default: {data_dir}/processed/{run_id}/readout)
- `--config`: Configuration file path (required)
- `--multi-channel`: Process as multi-channel (default: True)
- `--verbose, -v`: Verbose output for debugging

**Output Files:**
The pipeline generates the following output files in the `readout` directory:
- `intensity_raw.csv`: Raw intensity values (ch1, ch2, ch3, ch4)
- `intensity.csv`: Transformed intensity values (after matrix correction)
- `coordinates.csv`: Spot coordinates (Y, X)
- `metadata.yaml`: Processing metadata and parameters
- `channel_info.yaml`: Channel configuration and mapping information
- `full_config.yaml`: Complete merged configuration used
- `spot_detection.yaml`: Original configuration file copy

**Key Features:**
- **Unified Channel System**: Uses ch1, ch2, ch3, ch4 for consistent channel naming
- **Dynamic Memory Management**: Automatically calculates tiling thresholds based on available memory and channel count
- **Matrix-based Corrections**: Supports crosstalk correction and scaling transformations
- **Coordinate Deduplication**: Automatic removal of duplicate coordinates across channels
- **Large Image Support**: Automatic tiling for processing very large images
- **Modular Configuration**: Separate configuration files for different components

**Configuration System:**
The framework uses a modular configuration system:
- `spot_detection_base.yaml` - Base configuration (channels, memory, tiling)
- `spot_detection_traditional.yaml` - Traditional detection parameters
- `spot_detection_deep_learning.yaml` - Deep learning detection parameters  
- `spot_detection_intensity.yaml` - Intensity extraction parameters
- `spot_detection.yaml` - Main configuration file that loads all modules

**Dependencies:**
```
numpy
pandas
opencv-python
scikit-image
scipy
tqdm
pyyaml
stardist (for deep learning methods)
```

**Example Configuration:**
```yaml
# spot_detection_base.yaml
base:
  max_memory: 32  # GB
  use_tiling: true

channels:
  unified_channels: ['ch1', 'ch2', 'ch3', 'ch4']
  file_mapping:
    ch1: 'cyc_1_cy5.tif'    # 670nm - Cy5
    ch2: 'cyc_1_TxRed.tif'  # 615nm - Texas Red
    ch3: 'cyc_1_cy3.tif'    # 550nm - Cy3
    ch4: 'cyc_1_FAM.tif'    # 520nm - FAM
  wavelength_info:
    ch1:
      wavelength_nm: 670
      description: 'Cy5 - Far Red'
      original_name: 'cy5'
    # ... other channels
  transformation_matrix:
    ch1: [1.0, 0.0, 0.0, 0.0]
    ch2: [0.0, 1.0, 0.0, 0.0]
    ch3: [0.0, 0.0, 2.5, 0.0]
    ch4: [0.0, 0.0, -0.25, 0.75]
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

### Gene Calling Pipeline

#### `gene_calling.py`

Unified gene calling pipeline for signal point classification using GMM and other methods with modular configuration system.

**Usage:**
```bash
# Basic usage
python gene_calling.py \
    --data-dir /path/to/data \
    --run-id experiment_name

# With custom configuration
python gene_calling.py \
    --config code/configs/gene_calling.yaml \
    --data-dir /path/to/data \
    --run-id experiment_name \
    --output /path/to/output \
    --verbose

# With custom file names
python gene_calling.py \
    --data-dir /path/to/data \
    --run-id experiment_name \
    --intensity-file intensity.csv \
    --coordinates-file coordinates.csv
```

**Command Line Arguments:**
- `--config, -c`: Path to configuration file (default: configs/gene_calling.yaml)
- `--data-dir, -d`: Path to data directory containing intensity and coordinates files (required)
- `--run-id, -r`: Run ID for the experiment (required)
- `--output, -o`: Output directory (default: data_dir/run_id/readout/classification)
- `--intensity-file`: Intensity data filename (default: intensity.csv)
- `--coordinates-file`: Coordinates data filename (default: coordinates.csv)
- `--verbose, -v`: Enable verbose output

**Input Data Format:**
The pipeline expects data from the spot detection pipeline:
- **intensity.csv**: Contains columns `ch1, ch2, ch3, ch4` with corrected intensity values
- **coordinates.csv**: Contains columns `Y, X` with spot coordinates

Note: Crosstalk elimination and intensity scaling are already performed in spot detection.

**Output Files:**
The pipeline generates the following output files:
- `predictions.csv`: Classification results with predicted labels
- `trained_model.yaml`: Trained model parameters
- `config.yaml`: Copy of input configuration
- `full_config.yaml`: Complete processed configuration
- `evaluation_report.md`: Classification evaluation report
- `*.png`: Visualization plots (if enabled)
- `predictions_relabeled.csv`: Results after manual threshold adjustment (if enabled)
- `quantitative_evaluation.yaml`: Quantitative evaluation results (if enabled)

**Configuration System:**
The pipeline uses a modular YAML configuration system similar to spot detection:

**Main Configuration File:**
`configs/gene_calling.yaml` - Loads and combines configuration modules:
```yaml
config_modules:
  - gene_calling_base.yaml
  - gene_calling_gmm.yaml

overrides:
  # Override specific parameters if needed
  classification:
    gmm:
      num_per_layer: 20
```

**Configuration Modules:**

**`configs/gene_calling_base.yaml`** - Common parameters:
```yaml
data:
  input_format:
    intensity_columns: ["ch1", "ch2", "ch3", "ch4"]
    coordinate_columns: ["Y", "X"]

feature_extraction:
  feature_types: ["ratios", "projections", "intensity_features"]
  include_g_channel: true

evaluation:
  visualization:
    figure_size: [10, 8]
    dpi: 300
```

**`configs/gene_calling_gmm.yaml`** - GMM-specific parameters:
```yaml
preprocessing:
  fret_adjustments:
    G_ye_factor: 0.6
    B_g_factor: 0.1
  gaussian_noise_scale: 0.01

classification:
  method: "gmm"
  gmm:
    covariance_type: "diag"
    use_layers: true
    num_per_layer: 15
```

**Additional Features:**
```yaml
# Manual threshold adjustment
manual_threshold_adjustment:
  enabled: false
  mask_dir: "path/to/masks"

# Quantitative evaluation
quantitative_evaluation:
  enabled: false
  channels: ["Ye/A", "B/A", "R/A"]

# Data overview
data_overview:
  enabled: false
  sample_size: 10000
```

**Key Features:**
- **GMM-based Classification**: Gaussian Mixture Model for signal point classification
- **Layer-based Processing**: Support for multi-layer GMM classification
- **Modular Configuration**: Separate configuration files for different components
- **Integration with Spot Detection**: Direct use of spot detection output
- **Manual Threshold Adjustment**: Support for manual relabeling using mask images
- **Quantitative Evaluation**: Comprehensive evaluation metrics and quality control
- **Data Overview**: Automatic generation of data distribution plots

**Dependencies:**
```
numpy
pandas
scikit-learn
matplotlib
seaborn
pyyaml
tqdm
opencv-python
scipy
```

**Example Workflow:**
```bash
# 1. Run spot detection first
python spot_detection_pipeline.py \
    --input stitched \
    --run-id 20250915_iLock_activation_37 \
    --data-dir G:/spatial_data \
    --config code/configs/spot_detection.yaml

# 2. Run gene calling on spot detection output
python gene_calling.py \
    --data-dir "G:/spatial_data/processed/20250915_iLock_activation_37/readout" \
    --run-id "20250915_iLock_activation_37" \
    --config "code/configs/gene_calling.yaml" \
    --verbose
```

This will:
1. Load intensity and coordinates data from spot detection
2. Apply GMM-specific preprocessing (FRET adjustments, transformations)
3. Extract features for classification
4. Train GMM classifier
5. Make predictions
6. Evaluate results and generate visualizations
7. Save all results to the output directory

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