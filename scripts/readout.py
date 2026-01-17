"""
Readout pipeline for PRISM.

This pipeline performs:
1. Spot detection using spotiflow (or traditional methods) on stitched images
2. Intensity extraction using tophat+3x3max method
3. Deduplication across all channels
4. Outputs separate position.csv and intensity.csv files

Note: This pipeline does NOT perform scaling, renaming, or crosstalk correction.
Those operations should be done in the gene_calling pipeline.

Output files:
- position.csv: Y, X, index
- intensity.csv: index, and raw intensity values for all channels (original channel names)
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import tifffile
import logging
from concurrent.futures import ProcessPoolExecutor

# Import from installed PRISM package
# After running: pip install -e . (from code/ directory)
from src.readout.spot_detection import get_spot_coordinates
from src.readout.intensity_readout import read_intensity_tophat
from src.readout.deduplicate import deduplicate_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== Configuration ==========
# Constants - modify these for your dataset
CHANNELS = ['cy5', 'TxRed', 'cy3', 'FAM']  # PRISM channels
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'

# Paths
src_dir = BASE_DIR / RUN_ID
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
read_dir.mkdir(exist_ok=True)

# Detection parameters
DETECTION_METHOD = 'spotiflow'  # 'spotiflow', 'gaussian_tophat', or 'tophat'
DETECTION_SNR = {'cy5': 8.0, 'TxRed': 8.0, 'cy3': 8.0, 'FAM': 8.0}  # For traditional methods

# Intensity reading parameters
TOPHAT_RADIUS = 3
SEARCH_RADIUS = 1  # 3x3max (2*1+1 = 3)

# Deduplication parameters
DEDUP_THRESHOLD = 2  # Distance threshold for deduplication (pixels)


def _detect_spots_in_channel(args):
    """Helper function for parallel spot detection."""
    image_path, channel, method, method_kwargs = args
    try:
        image = tifffile.imread(str(image_path))
        coords = get_spot_coordinates(image, method=method, min_distance=2, **method_kwargs)
        return (channel, coords, None)
    except Exception as e:
        return (channel, np.empty((0, 2), dtype=np.float32), str(e))


def main(stc_dir=stc_dir, read_dir=read_dir):
    """
    Main function for readout pipeline.
    
    Parameters
    ----------
    stc_dir : Path
        Directory containing stitched images (cyc_1_channel.tif)
    read_dir : Path
        Output directory for readout results
    """
    logger.info("=" * 80)
    logger.info("PRISM Readout Pipeline")
    logger.info("=" * 80)
    logger.info(f"RUN_ID: {RUN_ID}")
    logger.info(f"Detection method: {DETECTION_METHOD}")
    logger.info(f"Intensity method: tophat+3x3max (tophat_radius={TOPHAT_RADIUS}, search_radius={SEARCH_RADIUS})")
    logger.info("=" * 80)
    
    # Stage 1: Spot Detection
    logger.info("=" * 80)
    logger.info("Stage 1: Spot Detection")
    logger.info("=" * 80)
    
    # Collect all channel images for detection
    detection_tasks = []
    for channel in CHANNELS:
        image_path = stc_dir / f'cyc_1_{channel}.tif'
        if not image_path.exists():
            logger.warning(f"Channel file not found: {image_path}, skipping...")
            continue
        
        # Prepare detection kwargs
        if DETECTION_METHOD == 'spotiflow':
            detection_kwargs = {
                'prob_thresh': 0.2,
                'device': 'cuda'  # or 'cpu'
            }
        else:
            detection_kwargs = {
                'snr': DETECTION_SNR.get(channel, 8.0),
                'tophat_radius': TOPHAT_RADIUS
            }
        
        detection_tasks.append((str(image_path), channel, DETECTION_METHOD, detection_kwargs))
    
    # Parallel spot detection
    all_coordinates = []
    channel_coords_dict = {}
    
    if detection_tasks:
        logger.info(f"Detecting spots in {len(detection_tasks)} channels...")
        with ProcessPoolExecutor(max_workers=min(len(detection_tasks), 4)) as executor:
            futures = [executor.submit(_detect_spots_in_channel, task) for task in detection_tasks]
            for future in tqdm(futures, desc='Detecting spots'):
                channel, coords, error = future.result()
                if error is None and len(coords) > 0:
                    channel_coords_dict[channel] = coords
                    all_coordinates.append(coords)
                    logger.info(f"  {channel}: {len(coords)} spots detected")
                elif error:
                    logger.warning(f"  {channel}: Error - {error}")
    
    if not all_coordinates:
        logger.warning("No spots detected in any channel!")
        return
    
    # Combine coordinates from all channels
    logger.info("Combining coordinates from all channels...")
    all_coords = np.vstack(all_coordinates)
    logger.info(f"Total coordinates before deduplication: {len(all_coords)}")
    
    # Remove exact duplicates (same pixel coordinates)
    coords_rounded = np.round(all_coords).astype(np.int32)
    _, unique_indices = np.unique(coords_rounded, axis=0, return_index=True)
    unique_coords = all_coords[unique_indices]
    logger.info(f"Total coordinates after removing exact duplicates: {len(unique_coords)}")
    
    # Stage 2: Intensity Reading
    logger.info("=" * 80)
    logger.info("Stage 2: Intensity Reading (tophat+3x3max)")
    logger.info("=" * 80)
    
    # Create intensity dataframe
    intensity_df = pd.DataFrame({'Y': unique_coords[:, 0], 'X': unique_coords[:, 1]})
    
    # Read intensities for each channel (keep original channel names)
    for channel in tqdm(CHANNELS, desc='Reading intensities'):
        image_path = stc_dir / f'cyc_1_{channel}.tif'
        if not image_path.exists():
            continue
        
        image = tifffile.imread(str(image_path))
        
        # Read intensities using tophat+3x3max
        intensities = read_intensity_tophat(
            image,
            unique_coords,
            tophat_radius=TOPHAT_RADIUS,
            search_radius=SEARCH_RADIUS
        )
        
        # Keep original channel names (no renaming)
        intensity_df[channel] = intensities
    
    logger.info(f"Intensity reading completed for {len(intensity_df)} spots")
    
    # Stage 3: Final Deduplication
    logger.info("=" * 80)
    logger.info("Stage 3: Final Deduplication")
    logger.info("=" * 80)
    
    n_before_dedup = len(intensity_df)
    
    # Calculate score for deduplication (use max intensity across channels)
    intensity_cols = [c for c in CHANNELS if c in intensity_df.columns]
    if intensity_cols:
        intensity_df['max_intensity'] = intensity_df[intensity_cols].max(axis=1)
        intensity_df = deduplicate_dataframe(
            intensity_df,
            coordinate_columns=['Y', 'X'],
            threshold=DEDUP_THRESHOLD,
            sort_by='max_intensity',
            ascending=False
        )
        intensity_df = intensity_df.drop(columns=['max_intensity'])
    else:
        intensity_df = deduplicate_dataframe(
            intensity_df,
            coordinate_columns=['Y', 'X'],
            threshold=DEDUP_THRESHOLD
        )
    
    logger.info(f"Deduplication: {n_before_dedup} -> {len(intensity_df)} spots")
    
    # Stage 4: Create Output Files
    logger.info("=" * 80)
    logger.info("Stage 4: Creating Output Files")
    logger.info("=" * 80)
    
    # Create index for matching position and intensity files
    intensity_df = intensity_df.reset_index(drop=True)
    intensity_df['index'] = intensity_df.index
    
    # Position file: Y, X, index
    position_df = intensity_df[['Y', 'X', 'index']].copy()
    position_file = read_dir / 'position.csv'
    position_df.to_csv(position_file, index=False)
    logger.info(f"Saved position file: {position_file} ({len(position_df)} spots)")
    
    # Intensity file: index, and all intensity columns (original channel names)
    intensity_cols = ['index'] + [c for c in intensity_df.columns 
                                  if c not in ['Y', 'X', 'index']]
    intensity_output_df = intensity_df[intensity_cols].copy()
    intensity_file = read_dir / 'intensity.csv'
    intensity_output_df.to_csv(intensity_file, index=False)
    logger.info(f"Saved intensity file: {intensity_file} ({len(intensity_output_df)} spots)")
    
    # Save parameters
    params = {
        'RUN_ID': RUN_ID,
        'CHANNELS': CHANNELS,
        'DETECTION_METHOD': DETECTION_METHOD,
        'TOPHAT_RADIUS': TOPHAT_RADIUS,
        'SEARCH_RADIUS': SEARCH_RADIUS,
        'DEDUP_THRESHOLD': DEDUP_THRESHOLD,
        'n_spots': len(intensity_df)
    }
    params_file = read_dir / 'readout_params.yaml'
    with open(params_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    logger.info(f"Saved parameters: {params_file}")
    
    logger.info("=" * 80)
    logger.info("Readout pipeline completed successfully!")
    logger.info(f"Output files:")
    logger.info(f"  - {position_file}")
    logger.info(f"  - {intensity_file}")
    logger.info("=" * 80)
    
    return position_df, intensity_output_df


if __name__ == '__main__':
    # Copy this file to the readout directory
    current_file_path = os.path.abspath(__file__)
    target_file_path = read_dir / os.path.basename(current_file_path)
    try:
        shutil.copy(current_file_path, target_file_path)
    except Exception as e:
        logger.warning(f"Could not copy script to readout directory: {e}")
    
    main(stc_dir=stc_dir, read_dir=read_dir)
