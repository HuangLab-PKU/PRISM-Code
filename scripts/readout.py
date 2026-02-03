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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import from installed PRISM package
from prism.readout.spot_detection import get_spot_coordinates
from prism.readout.intensity_readout import read_intensity_tophat
from prism.readout.utils import deduplicate_dataframe, block_starts


# Configure logging
_LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
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

# Log to file (same format as console)
_file_handler = logging.FileHandler(read_dir / 'readout.log', encoding='utf-8')
_file_handler.setFormatter(logging.Formatter(_LOG_FMT))
logging.getLogger().addHandler(_file_handler)

# Detection parameters
DETECTION_METHOD = 'spotiflow'  # 'spotiflow', 'gaussian_tophat', or 'tophat'
DETECTION_SNR = {'cy5': 3.0, 'TxRed': 3.0, 'cy3': 3.0, 'FAM': 3.0}  # For traditional methods

# Intensity reading parameters
TOPHAT_RADIUS = 3
SEARCH_RADIUS = 1  # 3x3max (2*1+1 = 3)

# Deduplication parameters
DEDUP_THRESHOLD = 2  # Distance threshold for deduplication (pixels)

# Block processing (one task = one block; small images become a single block)
BLOCK_SIZE = (2048, 2048)
BLOCK_OVERLAP = (64, 64)

# Workers
N_WORKERS = 8

def _read_intensity_in_block(args):
    """Worker: one block = one task. Receives block array (no memmap). Returns (channel, indices, intensities)."""
    (channel, block, indices, coords_local, tophat_radius, search_radius) = args
    if len(indices) == 0:
        return (channel, np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    intensities = read_intensity_tophat(
        block, coords_local,
        tophat_radius=tophat_radius, search_radius=search_radius
    )
    return (channel, indices, intensities)


def _detect_spots_in_block(args):
    """Worker: one block = one task. Receives block array (no memmap). Returns (channel, global_coords)."""
    channel, block, start_y, start_x, method, method_kwargs = args
    coords_local = get_spot_coordinates(block, method=method, min_distance=2, **method_kwargs)
    if len(coords_local) == 0:
        return (channel, np.empty((0, 2), dtype=np.float64), None)
    global_coords = coords_local + np.array([start_y, start_x], dtype=coords_local.dtype)
    return (channel, global_coords, None)


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
    
    # Spot detection: one memmap per channel (open once), read each block from it when needed, then discard block
    all_coordinates = []
    channel_coords_dict = {}

    def _detection_task_iter():
        for channel in CHANNELS:
            image_path = stc_dir / f'cyc_1_{channel}.tif'
            if DETECTION_METHOD == 'spotiflow':
                detection_kwargs = {'prob_thresh': None, 'device': 'cuda'}
            else:
                detection_kwargs = {
                    'snr': DETECTION_SNR.get(channel, 3.0),
                    'tophat_radius': TOPHAT_RADIUS
                }
            img = tifffile.memmap(str(image_path))
            if img.ndim == 3:
                img = img[0]
            h, w = img.shape
            by, bx = BLOCK_SIZE
            for start_y, start_x in block_starts(h, w, BLOCK_SIZE, BLOCK_OVERLAP):
                end_y = min(start_y + by, h)
                end_x = min(start_x + bx, w)
                block = np.asarray(img[start_y:end_y, start_x:end_x])
                yield (channel, block, start_y, start_x, DETECTION_METHOD, detection_kwargs)
            del img

    task_iter = _detection_task_iter()
    n_workers = N_WORKERS
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for _ in range(n_workers):
            task = next(task_iter, None)
            if task is None:
                break
            futures[executor.submit(_detect_spots_in_block, task)] = None
        n_blocks = 0
        results = []
        pbar = tqdm(desc='Detecting spots')
        while futures:
            for future in as_completed(futures):
                n_blocks += 1
                pbar.update(1)
                channel, coords, error = future.result()
                if error:
                    logger.warning(f"  {channel}: Error - {error}")
                elif len(coords) > 0:
                    results.append((channel, coords))
                del futures[future]
                task = next(task_iter, None)
                if task is not None:
                    futures[executor.submit(_detect_spots_in_block, task)] = None
                break
        pbar.close()

    if results:
        logger.info(f"Detecting spots: {n_blocks} blocks")
        by_channel = defaultdict(list)
        for channel, coords in results:
            by_channel[channel].append(coords)
        for channel in CHANNELS:
            if channel in by_channel:
                coords = np.vstack(by_channel[channel])
                channel_coords_dict[channel] = coords
                all_coordinates.append(coords)
                logger.info(f"  {channel}: {len(coords)} spots detected")
    
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
    
    # Stage 2: Intensity Reading (block + parallel, same as detection)
    logger.info("=" * 80)
    logger.info("Stage 2: Intensity Reading (tophat+3x3max, by blocks)")
    logger.info("=" * 80)

    intensity_df = pd.DataFrame({'Y': unique_coords[:, 0], 'X': unique_coords[:, 1]})
    margin = TOPHAT_RADIUS + SEARCH_RADIUS + 2  # margin for tophat + local max
    for channel in CHANNELS:
        intensity_df[channel] = np.nan

    # Intensity: one memmap per channel (open once), read each block from it when needed, then discard block
    def _intensity_task_iter():
        for channel in CHANNELS:
            image_path = stc_dir / f'cyc_1_{channel}.tif'
            img = tifffile.memmap(str(image_path))
            if img.ndim == 3:
                img = img[0]
            h, w = img.shape
            by, bx = BLOCK_SIZE
            for start_y, start_x in block_starts(h, w, BLOCK_SIZE, BLOCK_OVERLAP):
                end_y = min(start_y + by, h)
                end_x = min(start_x + bx, w)
                in_y = (unique_coords[:, 0] >= start_y) & (unique_coords[:, 0] < end_y)
                in_x = (unique_coords[:, 1] >= start_x) & (unique_coords[:, 1] < end_x)
                indices = np.where(in_y & in_x)[0]
                if len(indices) == 0:
                    continue
                y0 = max(0, start_y - margin)
                y1 = min(h, end_y + margin)
                x0 = max(0, start_x - margin)
                x1 = min(w, end_x + margin)
                block = np.asarray(img[y0:y1, x0:x1])
                coords_global = unique_coords[indices]
                coords_local = coords_global - np.array([y0, x0], dtype=coords_global.dtype)
                yield (channel, block, indices, coords_local, TOPHAT_RADIUS, SEARCH_RADIUS)
            del img

    intensity_task_iter = _intensity_task_iter()
    n_intensity_blocks = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        for _ in range(N_WORKERS):
            task = next(intensity_task_iter, None)
            if task is None:
                break
            futures[executor.submit(_read_intensity_in_block, task)] = None
        pbar_int = tqdm(desc='Reading intensities')
        while futures:
            for future in as_completed(futures):
                n_intensity_blocks += 1
                pbar_int.update(1)
                channel, indices, intensities = future.result()
                intensity_df.loc[indices, channel] = intensities
                del futures[future]
                task = next(intensity_task_iter, None)
                if task is not None:
                    futures[executor.submit(_read_intensity_in_block, task)] = None
                break
        pbar_int.close()
    logger.info(f"Reading intensities: {n_intensity_blocks} blocks")

    # Sanity: any NaN means a coord was not in any block (should not happen)
    for channel in CHANNELS:
        if intensity_df[channel].isna().any():
            n_miss = intensity_df[channel].isna().sum()
            logger.warning(f"  {channel}: {n_miss} spots had no intensity (fill with 0)")
            intensity_df[channel] = intensity_df[channel].fillna(0)
    intensity_df[CHANNELS] = intensity_df[CHANNELS].astype(np.float32)
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
        intensity_df = deduplicate_dataframe(intensity_df, coordinate_columns=['Y', 'X'], threshold=DEDUP_THRESHOLD)
    
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
    main(stc_dir=stc_dir, read_dir=read_dir)
