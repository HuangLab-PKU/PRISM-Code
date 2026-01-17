"""
Gene calling pipeline for PRISM.

This pipeline performs signal point classification:
- Input: intensity.csv (from readout.py, with original channel names: cy5, TxRed, cy3, FAM)
- Output: mapping.csv (gene assignments with index matching position.csv and intensity.csv)

This pipeline handles:
- Channel renaming (cy5->R, TxRed->Ye, cy3->G, FAM->B)
- Crosstalk correction
- Intensity scaling
- Feature extraction and classification

The script uses the existing gene_calling pipeline from src/gene_calling.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# Add PRISM code path
package_path = 'path_to_PRISM_code_src'
if package_path not in sys.path:
    sys.path.append(package_path)

from gene_calling.pipeline import SignalClassificationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'
PRISM_PANEL = 'PRISM30'

# Paths
src_dir = BASE_DIR / RUN_ID
read_dir = src_dir / 'readout'
config_dir = Path('path_to_configs')  # Directory containing gene_calling configs

# Input/Output files
INTENSITY_FILE = read_dir / 'intensity.csv'
POSITION_FILE = read_dir / 'position.csv'  # Optional, for validation
MAPPING_FILE = read_dir / 'mapping.csv'
CONFIG_FILE = config_dir / f'gene_calling_{PRISM_PANEL.lower()}.yaml'  # Optional config file


def main(intensity_file=INTENSITY_FILE, position_file=POSITION_FILE, 
         mapping_file=MAPPING_FILE, config_file=None, PRISM_PANEL='PRISM30'):
    """
    Main function for gene calling.
    
    Parameters
    ----------
    intensity_file : Path
        Path to intensity.csv file (from readout.py, with original channel names)
    position_file : Path, optional
        Path to position.csv file (for validation, not required)
    mapping_file : Path
        Path to output mapping.csv file
    config_file : Path, optional
        Path to YAML configuration file. If None, uses default config.
    PRISM_PANEL : str
        PRISM panel type (PRISM30, PRISM45, etc.)
    """
    logger.info("=" * 80)
    logger.info("PRISM Gene Calling Pipeline")
    logger.info("=" * 80)
    logger.info(f"RUN_ID: {RUN_ID}")
    logger.info(f"PRISM_PANEL: {PRISM_PANEL}")
    logger.info(f"Input: {intensity_file}")
    logger.info(f"Output: {mapping_file}")
    logger.info("=" * 80)
    
    # Check input file
    if not intensity_file.exists():
        raise FileNotFoundError(f"Intensity file not found: {intensity_file}")
    
    # Load intensity data
    logger.info("Loading intensity data...")
    intensity_df = pd.read_csv(intensity_file)
    logger.info(f"Loaded {len(intensity_df)} signal points")
    
    # Validate that index column exists
    if 'index' not in intensity_df.columns:
        logger.warning("'index' column not found in intensity file. Creating index...")
        intensity_df['index'] = intensity_df.index
    
    # Stage 0: Channel renaming and preprocessing
    logger.info("=" * 80)
    logger.info("Stage 0: Channel Renaming and Preprocessing")
    logger.info("=" * 80)
    
    # Map channel names from readout output to PRISM standard names
    channel_map = {'cy5': 'R', 'TxRed': 'Ye', 'cy3': 'G', 'FAM': 'B'}
    for old_ch, new_ch in channel_map.items():
        if old_ch in intensity_df.columns:
            intensity_df[new_ch] = intensity_df[old_ch]
            intensity_df = intensity_df.drop(columns=[old_ch])
            logger.info(f"Renamed channel: {old_ch} -> {new_ch}")
    
    # Crosstalk elimination
    if 'B' in intensity_df.columns and 'G' in intensity_df.columns:
        intensity_df['B'] = intensity_df['B'] - intensity_df['G'] * 0.25
        intensity_df['B'] = np.maximum(intensity_df['B'], 0)
        logger.info("Applied crosstalk correction: B = B - G * 0.25")
    
    # Scale intensities
    if 'R' in intensity_df.columns:
        intensity_df['Scaled_R'] = intensity_df['R']
    if 'Ye' in intensity_df.columns:
        intensity_df['Scaled_Ye'] = intensity_df['Ye']
    if 'G' in intensity_df.columns:
        intensity_df['Scaled_G'] = intensity_df['G'] * 2.5
    if 'B' in intensity_df.columns:
        intensity_df['Scaled_B'] = intensity_df['B'] * 0.75
    logger.info("Applied intensity scaling: R=1.0, Ye=1.0, G=2.5, B=0.75")
    
    # Calculate sum for thresholding
    intensity_cols = [c for c in ['Scaled_R', 'Scaled_Ye', 'Scaled_B'] if c in intensity_df.columns]
    if intensity_cols:
        intensity_df['sum'] = intensity_df[intensity_cols].sum(axis=1)
        
        # Apply threshold based on PRISM panel
        SUM_THRESHOLD = 800
        n_before = len(intensity_df)
        if PRISM_PANEL in ('PRISM30', 'PRISM45', 'PRISM63'):
            intensity_df = intensity_df[intensity_df['sum'] >= SUM_THRESHOLD]
        elif PRISM_PANEL in ('PRISM31', 'PRISM46', 'PRISM64'):
            G_THRESHOLD = 3
            G_ABS_THRESHOLD = 1000
            mask = (intensity_df['sum'] >= SUM_THRESHOLD) | \
                   ((intensity_df.get('Scaled_G', 0) / intensity_df['sum'] >= G_THRESHOLD) & 
                    (intensity_df.get('Scaled_G', 0) > G_ABS_THRESHOLD))
            intensity_df = intensity_df[mask]
        
        logger.info(f"Applied intensity threshold: {n_before} -> {len(intensity_df)} spots")
    
    # Load position file for validation (optional)
    if position_file and position_file.exists():
        logger.info("Loading position data for validation...")
        position_df = pd.read_csv(position_file)
        logger.info(f"Loaded {len(position_df)} positions")
        
        # Validate that indices match
        intensity_indices = set(intensity_df['index'].values)
        position_indices = set(position_df['index'].values)
        if intensity_indices != position_indices:
            logger.warning(f"Index mismatch: intensity has {len(intensity_indices)} indices, "
                         f"position has {len(position_indices)} indices")
            # Use intersection
            common_indices = intensity_indices & position_indices
            intensity_df = intensity_df[intensity_df['index'].isin(common_indices)]
            logger.info(f"Using {len(common_indices)} common indices")
    
    # Initialize pipeline
    logger.info("Initializing classification pipeline...")
    if config_file and Path(config_file).exists():
        logger.info(f"Loading config from: {config_file}")
        pipeline = SignalClassificationPipeline(config_path=config_file)
    else:
        logger.info("Using default configuration")
        # Create default config based on PRISM panel
        default_config = _get_default_config(PRISM_PANEL)
        pipeline = SignalClassificationPipeline(config=default_config)
    
    # Fit pipeline on data (unsupervised learning)
    logger.info("=" * 80)
    logger.info("Stage 1: Fitting Classification Model")
    logger.info("=" * 80)
    pipeline.fit(intensity_df)
    logger.info("Model fitting completed")
    
    # Make predictions
    logger.info("=" * 80)
    logger.info("Stage 2: Making Predictions")
    logger.info("=" * 80)
    result = pipeline.predict(intensity_df)
    logger.info(f"Predicted labels for {len(result.labels)} samples")
    
    # Create mapping dataframe
    logger.info("=" * 80)
    logger.info("Stage 3: Creating Mapping File")
    logger.info("=" * 80)
    
    # Get mapping from result
    mapping_df = result.to_dataframe(intensity_df)
    
    # Ensure index column is preserved and matches intensity_df
    if 'index' in intensity_df.columns:
        # Use the index from intensity_df to ensure perfect matching
        mapping_df['index'] = intensity_df['index'].values
    else:
        # If no index column, create one
        mapping_df['index'] = mapping_df.index
        logger.warning("No 'index' column found in intensity file. Created new index.")
    
    # Reorder columns to put index first
    cols = ['index'] + [c for c in mapping_df.columns if c != 'index']
    mapping_df = mapping_df[cols]
    
    # Validate index matching
    if 'index' in intensity_df.columns:
        intensity_indices = set(intensity_df['index'].values)
        mapping_indices = set(mapping_df['index'].values)
        if intensity_indices != mapping_indices:
            logger.warning(f"Index mismatch detected: intensity has {len(intensity_indices)} indices, "
                         f"mapping has {len(mapping_indices)} indices")
            # Use intersection to ensure consistency
            common_indices = sorted(list(intensity_indices & mapping_indices))
            mapping_df = mapping_df[mapping_df['index'].isin(common_indices)].sort_values('index')
            logger.info(f"Using {len(common_indices)} common indices for mapping file")
    
    # Save mapping file
    mapping_df.to_csv(mapping_file, index=False)
    logger.info(f"Saved mapping file: {mapping_file} ({len(mapping_df)} entries)")
    
    # Generate summary statistics
    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    if 'predicted_label' in mapping_df.columns or 'gene' in mapping_df.columns:
        label_col = 'predicted_label' if 'predicted_label' in mapping_df.columns else 'gene'
        label_counts = mapping_df[label_col].value_counts()
        logger.info(f"Number of unique labels: {len(label_counts)}")
        logger.info(f"Top 10 labels:")
        for label, count in label_counts.head(10).items():
            logger.info(f"  {label}: {count}")
    
    logger.info("=" * 80)
    logger.info("Gene calling completed successfully!")
    logger.info(f"Output file: {mapping_file}")
    logger.info("=" * 80)
    
    return mapping_df


def _get_default_config(PRISM_PANEL='PRISM30'):
    """Get default configuration based on PRISM panel type."""
    # Panel-specific parameters
    panel_configs = {
        'PRISM30': {
            'num_per_layer': 15,
            'g_layer_num': 2,
            'total_components': 30,
            'color_grade': 5,
            'layer_grade': 2,
        },
        'PRISM45': {
            'num_per_layer': 15,
            'g_layer_num': 3,
            'total_components': 45,
            'color_grade': 5,
            'layer_grade': 3,
        },
        'PRISM63': {
            'num_per_layer': 21,
            'g_layer_num': 3,
            'total_components': 63,
            'color_grade': 7,
            'layer_grade': 3,
        },
        'PRISM64': {
            'num_per_layer': 21,
            'g_layer_num': 3,
            'total_components': 64,
            'color_grade': 7,
            'layer_grade': 3,
        },
    }
    
    panel_params = panel_configs.get(PRISM_PANEL, panel_configs['PRISM30'])
    
    config = {
        "preprocessing": {
            "scaling_factors": {"R": 1.0, "Ye": 1.0, "G": 2.5, "B": 0.75},
            "crosstalk_factor": 0.25,
            "fret_adjustments": {"G_ye_factor": 0.6, "B_g_factor": 0.1},
            "gaussian_noise_scale": 0.01,
            "prism_panel": PRISM_PANEL,
            "thre_min": 200,
            "thre_max": 10000,
        },
        "feature_extraction": {
            "feature_types": ["ratios", "projections", "intensity_features"],
            "include_g_channel": True,
        },
        "classification": {
            "method": "gmm",
            "gmm": {
                "covariance_type": "diag",
                "max_iter": 100,
                "tol": 1e-3,
                "n_init": 1,
                "use_layers": True,
                "g_layer_column": "G_layer",
                "num_per_layer": panel_params['num_per_layer'],
                "scale_features": True,
            },
        },
        "prism_panel": {
            "type": PRISM_PANEL,
            "num_per_layer": panel_params['num_per_layer'],
            "g_layer_num": panel_params['g_layer_num'],
            "total_components": panel_params['total_components'],
            "channel_grading": {
                "color_channels": panel_params['color_grade'],
                "layer_channel": panel_params['layer_grade'],
            },
        },
        "evaluation": {
            "visualization": {
                "figure_size": (10, 8),
                "dpi": 300
            }
        },
    }
    
    return config


if __name__ == '__main__':
    # Copy this file to the readout directory
    current_file_path = os.path.abspath(__file__)
    target_file_path = read_dir / os.path.basename(current_file_path)
    try:
        shutil.copy(current_file_path, target_file_path)
    except Exception as e:
        logger.warning(f"Could not copy script to readout directory: {e}")
    
    # Run main function
    config_file = CONFIG_FILE if CONFIG_FILE.exists() else None
    main(
        intensity_file=INTENSITY_FILE,
        position_file=POSITION_FILE,
        mapping_file=MAPPING_FILE,
        config_file=config_file,
        PRISM_PANEL=PRISM_PANEL
    )
