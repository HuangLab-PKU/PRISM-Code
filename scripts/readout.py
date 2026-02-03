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


# Log format (used for both console and file)
_LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logger = logging.getLogger(__name__)

# ========== Configuration (set from config file + CLI) ==========
CHANNELS = []  # channel labels (file stems) for output columns and logging
CHANNEL_FILES = []  # filenames under stitched dir; path = stc_dir / fname
BASE_DIR = None
RUN_ID = None
src_dir = None
stc_dir = None
read_dir = None
DETECTION_METHOD = 'spotiflow'
DETECTION_SNR = {}
TOPHAT_RADIUS = 3
SEARCH_RADIUS = 1
DEDUP_THRESHOLD = 2
BLOCK_SIZE = (2048, 2048)
BLOCK_OVERLAP = (64, 64)
N_WORKERS = 4


def load_config(config_dir):
    """Load readout.yaml from config_dir and set module-level config globals."""
    global CHANNELS, CHANNEL_FILES, BASE_DIR, DETECTION_METHOD, DETECTION_SNR
    global TOPHAT_RADIUS, SEARCH_RADIUS, DEDUP_THRESHOLD
    global BLOCK_SIZE, BLOCK_OVERLAP, N_WORKERS
    config_path = Path(config_dir) / 'readout.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    CHANNEL_FILES = [str(f) for f in cfg['channel_files']]
    CHANNELS = [Path(f).stem for f in CHANNEL_FILES]

    BASE_DIR = Path(cfg.get('base_dir', '.'))
    DETECTION_METHOD = cfg.get('detection_method', DETECTION_METHOD)
    DETECTION_SNR = cfg.get('detection_snr', DETECTION_SNR)
    TOPHAT_RADIUS = cfg.get('tophat_radius', TOPHAT_RADIUS)
    SEARCH_RADIUS = cfg.get('search_radius', SEARCH_RADIUS)
    DEDUP_THRESHOLD = cfg.get('dedup_threshold', DEDUP_THRESHOLD)
    bs = cfg.get('block_size', list(BLOCK_SIZE))
    BLOCK_SIZE = tuple(bs) if isinstance(bs, list) else bs
    bo = cfg.get('block_overlap', list(BLOCK_OVERLAP))
    BLOCK_OVERLAP = tuple(bo) if isinstance(bo, list) else bo
    N_WORKERS = int(cfg.get('n_workers', N_WORKERS))


def _read_intensity_in_block(args):
    """Worker: one block = one task. Receives block array (no memmap). Returns (channel, indices, intensities)."""
    (channel, block, indices, coords_local, tophat_radius, search_radius) = args
    if len(indices) == 0:
        return (channel, np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    try:
        intensities = read_intensity_tophat(
            block, coords_local,
            tophat_radius=tophat_radius, search_radius=search_radius
        )
        return (channel, indices, intensities)
    except Exception as e:
        return (channel, indices, np.full(len(indices), np.nan, dtype=np.float64))


def _detect_spots_in_block(args):
    """Worker: one block = one task. Receives block array (no memmap). Returns (channel, global_coords, error)."""
    channel, block, start_y, start_x, method, method_kwargs = args
    try:
        coords_local = get_spot_coordinates(block, method=method, min_distance=2, **method_kwargs)
        if len(coords_local) == 0:
            return (channel, np.empty((0, 2), dtype=np.float64), None)
        global_coords = coords_local + np.array([start_y, start_x], dtype=coords_local.dtype)
        return (channel, global_coords, None)
    except Exception as e:
        return (channel, np.empty((0, 2), dtype=np.float64), str(e))


def main(run_id, stc_dir, read_dir):
    """
    Main function for readout pipeline.

    Parameters
    ----------
    run_id : str
        Run identifier (e.g. 20260201_WXC_ZJX_CRC_patient1_5um_2)
    stc_dir : Path
        Directory containing stitched images (cyc_1_channel.tif)
    read_dir : Path
        Output directory for readout results
    """
    logger.info("=" * 80)
    logger.info("PRISM Readout Pipeline")
    logger.info("=" * 80)
    logger.info(f"RUN_ID: {run_id}")
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

    # Count total detection blocks for progress bar (use TiffFile for shape only)
    n_detection_total = 0
    for _fname in CHANNEL_FILES:
        with tifffile.TiffFile(stc_dir / _fname) as _t:
            _sh = _t.pages[0].shape
        _h, _w = (_sh[0], _sh[1]) if len(_sh) == 2 else (_sh[1], _sh[2])
        n_detection_total += len(block_starts(_h, _w, BLOCK_SIZE, BLOCK_OVERLAP))

    def _detection_task_iter():
        for channel, fname in zip(CHANNELS, CHANNEL_FILES):
            image_path = stc_dir / fname
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
        pbar = tqdm(desc='Detecting spots', total=n_detection_total)
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

    # Precompute which coords fall in which block (one pass over unique_coords per channel; no repeated scan)
    by, bx = BLOCK_SIZE
    oy, ox = BLOCK_OVERLAP
    step_y = max(1, by - oy)
    step_x = max(1, bx - ox)
    coords_in_block = defaultdict(dict)  # channel -> (start_y, start_x) -> np.ndarray of indices
    for channel in CHANNELS:
        bucket = defaultdict(list)
        for i in range(len(unique_coords)):
            y, x = float(unique_coords[i, 0]), float(unique_coords[i, 1])
            start_y = step_y * (int(y) // step_y)
            start_x = step_x * (int(x) // step_x)
            bucket[(start_y, start_x)].append(i)
        for key, idx_list in bucket.items():
            coords_in_block[channel][key] = np.array(idx_list, dtype=np.int64)
    n_intensity_total = sum(1 for ch in CHANNELS for idx in coords_in_block.get(ch, {}).values() if len(idx) > 0)

    # Intensity: one memmap per channel (open once), read each block from it when needed, then discard block
    def _intensity_task_iter():
        for channel, fname in zip(CHANNELS, CHANNEL_FILES):
            image_path = stc_dir / fname
            img = tifffile.memmap(str(image_path))
            if img.ndim == 3:
                img = img[0]
            h, w = img.shape
            by, bx = BLOCK_SIZE
            ch_blocks = coords_in_block.get(channel, {})
            for start_y, start_x in block_starts(h, w, BLOCK_SIZE, BLOCK_OVERLAP):
                indices = ch_blocks.get((start_y, start_x), np.array([], dtype=np.int64))
                if len(indices) == 0:
                    continue
                y0 = max(0, start_y - margin)
                y1 = min(h, min(start_y + by, h) + margin)
                x0 = max(0, start_x - margin)
                x1 = min(w, min(start_x + bx, w) + margin)
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
        pbar_int = tqdm(desc='Reading intensities', total=n_intensity_total)
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
    position_df = intensity_df[['index', 'Y', 'X']].copy()
    position_file = read_dir / 'position.csv'
    position_df.to_csv(position_file, index=False)
    logger.info(f"Saved position file: {position_file} ({len(position_df)} spots)")
    
    # Intensity file: index, and all intensity columns (original channel names)
    intensity_cols = ['index'] + [c for c in intensity_df.columns if c not in ['Y', 'X', 'index']]
    intensity_output_df = intensity_df[intensity_cols].copy()
    intensity_file = read_dir / 'intensity.csv'
    intensity_output_df.to_csv(intensity_file, index=False)
    logger.info(f"Saved intensity file: {intensity_file} ({len(intensity_output_df)} spots)")
    
    # Save parameters
    params = {
        'RUN_ID': run_id,
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
    import argparse
    _script_dir = Path(__file__).resolve().parent
    _default_config_dir = _script_dir.parent / 'configs'
    _parser = argparse.ArgumentParser(description='PRISM readout: spot detection + intensity')
    _parser.add_argument('run_id', help='RUN_ID (e.g. 20260201_WXC_ZJX_CRC_patient1_5um_2)')
    _parser.add_argument('--config-dir', type=Path, default=_default_config_dir,
                         help=f'Directory containing readout.yaml (default: {_default_config_dir})')
    _args = _parser.parse_args()

    load_config(_args.config_dir)
    _src_dir = BASE_DIR / f'{_args.run_id}_processed'
    _stc_dir = _src_dir / 'stitched'
    _read_dir = _src_dir / 'readout'
    _read_dir.mkdir(exist_ok=True)

    # Log to file (same format as console)
    _root = logging.getLogger()
    _fh = logging.FileHandler(_read_dir / 'readout.log', encoding='utf-8')
    _fh.setFormatter(logging.Formatter(_LOG_FMT))
    _root.addHandler(_fh)

    main(run_id=_args.run_id, stc_dir=_stc_dir, read_dir=_read_dir)
