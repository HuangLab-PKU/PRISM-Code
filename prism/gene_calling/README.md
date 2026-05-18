# Signal Classification Framework

A flexible and extensible framework for classifying signal points based on multi-channel intensity values. This framework is designed specifically for PRISM (Probe-based Imaging for Sequential Multiplexing) data analysis.

## Features

- **Multi-channel signal classification**: Handles 4-channel intensity data (ch1-ch4)
- **Flexible preprocessing**: Crosstalk elimination, intensity scaling, FRET adjustments
- **Multiple classification methods**: Currently supports GMM, easily extensible
- **Layer-based classification**: Supports G-layer stratification for better clustering
- **Comprehensive evaluation**: Multiple metrics and visualizations
- **Configurable pipeline**: YAML-based configuration system

## Architecture

The framework consists of several key components:

### Core Components

1. **Base Classes** (`base.py`)
   - `BaseClassifier`: Abstract base class for all classifiers
   - `BasePreprocessor`: Abstract base class for data preprocessing
   - `BaseFeatureExtractor`: Abstract base class for feature extraction
   - `ClassificationResult`: Container for classification results

2. **Data Processing** (`data_processor.py`)
   - `SignalDataProcessor`: Handles preprocessing pipeline
   - `SignalFeatureExtractor`: Extracts features from preprocessed data

3. **Classification Methods** (`gmm_classifier.py`)
   - `GMMClassifier`: Gaussian Mixture Model implementation

4. **Evaluation** (`evaluator.py`)
   - `ClassificationEvaluator`: Comprehensive evaluation and visualization

5. **Pipeline** (`pipeline.py`)
   - `SignalClassificationPipeline`: Complete end-to-end pipeline

## Signal Encoding Scheme

The framework is designed around the following signal encoding scheme:

- **Three channel ratios**: ch1/sum, ch2/sum, ch4/sum (constrained to sum to 1)
- **Fourth channel relative intensity**: ch3/sum (independent)
- **Sum intensity**: Long-tail distribution along sum direction
- **3D space clustering**: Ratios form clusters on a 2D plane in 3D space
- **Gaussian distributions**: ch3/sum shows 1-3 Gaussian distributions

## Quick Start

### Basic Usage

```python
from signal_classification import SignalClassificationPipeline

# Initialize pipeline with configuration
pipeline = SignalClassificationPipeline(config_path='configs/signal_classification.yaml')

# Run complete pipeline
results = pipeline.run_full_pipeline(
    data_path='path/to/intensity.csv',
    coordinates_path='path/to/coordinates.csv',
    output_dir='results/'
)
```

### Step-by-Step Usage

```python
# 1. Initialize pipeline
pipeline = SignalClassificationPipeline(config_path='config.yaml')

# 2. Load data
data = pipeline.load_data('intensity.csv', 'coordinates.csv')

# 3. Fit classifier
pipeline.fit(data)

# 4. Make predictions
result = pipeline.predict(data)

# 5. Evaluate results
evaluation = pipeline.evaluate(result, data, output_dir='results/')
```

## Configuration

The framework uses YAML configuration files. Key configuration sections:

### Preprocessing
```yaml
preprocessing:
  scaling_factors:
    ch1: 1.0  # R channel (Cy5)
    ch2: 1.0  # Ye channel (TxRed)
    ch3: 1.0  # B channel (Cy3)
    ch4: 1.0  # G channel (FAM)
  crosstalk_factor: 0.25
  thre_min: 200
  thre_max: 10000
```

### Classification
```yaml
classification:
  method: "gmm"
  gmm:
    covariance_type: "diag"
    use_layers: true
    num_per_layer: 15
```

### Feature Extraction
```yaml
feature_extraction:
  feature_types:
    - "ratios"
    - "projections"
    - "intensity_features"
  include_g_channel: true
```

## Data Format

### Input Data
- **Intensity CSV**: Columns `ch1`, `ch2`, `ch3`, `ch4` with intensity values
- **Coordinates CSV**: Columns `Y`, `X` with spatial coordinates (optional)

### Output Data
- **Predictions CSV**: Original data + `predicted_label` + `prediction_confidence`
- **Visualizations**: Feature space plots, cluster distributions, etc.
- **Evaluation Report**: Markdown report with metrics and analysis

## Extending the Framework

### Adding New Classification Methods

1. Create a new classifier class inheriting from `BaseClassifier`
2. Implement required methods: `fit()`, `predict()`, `get_feature_importance()`
3. Add configuration options in the YAML file
4. Update the pipeline to support the new method

Example:
```python
class KMeansClassifier(BaseClassifier):
    def fit(self, features, labels=None):
        # Implementation
        return self
    
    def predict(self, features):
        # Implementation
        return ClassificationResult(...)
```

### Adding New Feature Extractors

1. Create a new extractor class inheriting from `BaseFeatureExtractor`
2. Implement `extract_features()` and `get_feature_names()`
3. Add configuration options

### Adding New Preprocessors

1. Create a new preprocessor class inheriting from `BasePreprocessor`
2. Implement `preprocess()` method
3. Add validation in `validate_data()` if needed

## Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Cluster Quality**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index
- **Supervised Metrics**: Adjusted Rand index, Normalized Mutual Information (if ground truth available)
- **Visualizations**: Feature space plots, cluster size distributions, confidence distributions
- **Statistical Analysis**: Feature distributions across clusters

## Examples

See `code/scripts/signal_classification_example.py` for a complete working example using your data format.

## Integration with PRISM Pipeline

The framework is designed to integrate seamlessly with the existing PRISM pipeline:

1. **Input**: Uses standard PRISM intensity and coordinate files
2. **Configuration**: Follows PRISM configuration patterns
3. **Output**: Generates results compatible with downstream PRISM analysis
4. **Visualization**: Uses PRISM color schemes and plotting conventions

## Performance Considerations

- **Memory**: Large datasets are handled efficiently with optional feature scaling
- **Speed**: GMM with diagonal covariance is optimized for speed
- **Scalability**: Framework supports batch processing and parallel execution
- **Storage**: Results are saved in efficient formats (CSV, YAML)

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce dataset size or enable feature scaling
2. **Poor clustering**: Adjust intensity thresholds or GMM parameters
3. **Missing features**: Check that all required columns are present in input data
4. **Configuration errors**: Validate YAML syntax and parameter values

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Additional classification methods (K-means, DBSCAN, etc.)
- Deep learning-based classification
- Real-time classification capabilities
- Advanced visualization options
- Integration with other PRISM modules
