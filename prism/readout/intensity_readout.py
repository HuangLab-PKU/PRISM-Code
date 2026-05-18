"""Intensity readout module for extracting intensity values at detected coordinates."""

import numpy as np
import pandas as pd
import cv2
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import maximum_filter
from skimage.io import imread

logger = logging.getLogger('readout_intensity')


def read_intensity_tophat(image, coordinates, tophat_radius=3, search_radius=1):
    """Read intensity values from top-hat filtered image with local peak search.
    
    This method applies top-hat morphological filtering directly to the raw image,
    and then searches for the maximum intensity within a local window around each
    coordinate. This provides robustness against small registration errors (drift).
    
    This is the recommended method for PRISM: tophat + 3x3max (search_radius=1).
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float, will be rounded)
    tophat_radius : int
        Top-hat morphological operation radius (default: 3)
    search_radius : int
        Radius to search for local maximum (default: 1, i.e., 3x3 window).
        Set to 0 to disable search (read center pixel only).
        
    Returns
    -------
    np.ndarray
        (N,) array of intensity values
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)
    
    # Apply top-hat filtering (no Gaussian blur)
    ksz = 2 * int(tophat_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    tophat_image = cv2.morphologyEx(image_f, cv2.MORPH_TOPHAT, kernel)
    
    # Apply local maximum search if requested (vectorized)
    # search_radius=1 means 3x3 window (2*1+1 = 3)
    if search_radius > 0:
        size = 2 * int(search_radius) + 1
        # maximum_filter is equivalent to local max search in window
        # This is much faster than looping over points for large N
        image_to_read = maximum_filter(tophat_image, size=size, mode='constant', cval=0)
    else:
        image_to_read = tophat_image
    
    coords = np.round(coordinates).astype(np.int32)
    height, width = image.shape
    
    # Clip coordinates to be safe
    y = np.clip(coords[:, 0], 0, height - 1)
    x = np.clip(coords[:, 1], 0, width - 1)
    
    return image_to_read[y, x]


def read_intensity_raw(image, coordinates, search_radius=0):
    """Read intensity values directly from raw image at specified coordinates.
    
    Parameters
    ----------
    image : np.ndarray
        Raw image (uint16)
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates (can be float, will be rounded)
    search_radius : int
        Radius to search for local maximum (default: 0).
        If > 0, reads the maximum value within the radius.
        
    Returns
    -------
    np.ndarray
        (N,) array of intensity values
    """
    # Convert to float32 for processing
    image_f = image.astype(np.float32, copy=False)
    
    # Apply local maximum search if requested (vectorized)
    if search_radius > 0:
        size = 2 * int(search_radius) + 1
        # maximum_filter is equivalent to local max search in window
        image_to_read = maximum_filter(image_f, size=size, mode='constant', cval=0)
    else:
        image_to_read = image_f

    coords = np.round(coordinates).astype(np.int32)
    height, width = image.shape
    
    # Clip coordinates to be safe
    y = np.clip(coords[:, 0], 0, height - 1)
    x = np.clip(coords[:, 1], 0, width - 1)
    
    return image_to_read[y, x]


# Dictionary mapping method names to functions
INTENSITY_READ_METHODS = {
    'raw': read_intensity_raw,
    'tophat': read_intensity_tophat,
}


def _read_intensity_for_single_image(args):
    """Helper function for parallel intensity reading (must be at module level for pickle).
    
    Parameters
    ----------
    args : tuple
        (tile_path, channel, cyc, coordinates, intensity_method, method_kwargs)
        tile_path is a string path to the image file
        coordinates is a numpy array of (Y, X) coordinates
        intensity_method is the method name ('raw', 'tophat', etc.)
        method_kwargs is a dict of additional parameters for the read method
        
    Returns
    -------
    tuple
        (cyc, channel, intensities, error_message)
        intensities is a numpy array or None if error
    """
    tile_path, channel, cyc, coordinates, intensity_method, method_kwargs = args
    try:
        # Load raw image if path provided
        raw_image = imread(str(tile_path))
        
        # Get the read function
        read_func = INTENSITY_READ_METHODS.get(intensity_method)
        if read_func is None:
            return (cyc, channel, None, f"Unknown intensity method: {intensity_method}")
        
        # Read intensities using specified method
        intensities = read_func(raw_image, coordinates, **method_kwargs)
        return (cyc, channel, intensities, None)
    except Exception as e:
        return (cyc, channel, None, str(e))


def get_intensity_df_for_tile(registered_dir, tile_name, coordinates, channels, cyc_num, 
                               feature_cache=None, max_workers=8, intensity_method='tophat', **method_kwargs):
    """Build an intensity dataframe for provided coordinates in a single tile using parallel processing.
    
    Parameters
    ----------
    registered_dir : str or Path
        Base directory containing cyc_n_chn folders or stitched image files
    tile_name : str
        Tile file name pattern (e.g., 'cyc_1_cy3.tif' for stitched images)
        For PRISM, this is typically the stitched image filename
    coordinates : np.ndarray
        Nx2 array-like of (Y,X) coordinates in tile-local coordinates (can be float)
    channels : list
        List of channel names
    cyc_num : int
        Number of cycles to read
    feature_cache : dict, optional
        Cache of feature images (not used for intensity reading, kept for compatibility)
    max_workers : int
        Maximum number of parallel workers for intensity reading (default: 8)
    intensity_method : str
        Intensity reading method: 'raw', 'tophat' (default: 'tophat')
    **method_kwargs
        Additional parameters for intensity reading method
        For 'tophat': tophat_radius (default: 3), search_radius (default: 1 for 3x3max)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Y','X', 'cyc_1_ch1', 'cyc_1_ch2', ...]
    """
    registered_dir = Path(registered_dir)
    coords = np.asarray(coordinates)
    
    # If no coordinates, return an empty dataframe with the expected columns
    cols = ['Y', 'X'] + [f'cyc_{c}_{ch}' for c in range(1, cyc_num + 1) for ch in channels]
    if coords.size == 0:
        return pd.DataFrame(columns=cols)

    intensity_df = pd.DataFrame({'Y': coords[:, 0], 'X': coords[:, 1]})
    
    # Get the read function
    read_func = INTENSITY_READ_METHODS.get(intensity_method)
    if read_func is None:
        raise ValueError(f"Unknown intensity method: {intensity_method}")
    
    # Set default kwargs for tophat method (3x3max)
    if intensity_method == 'tophat':
        default_kwargs = {
            'tophat_radius': method_kwargs.get('tophat_radius', 3),
            'search_radius': method_kwargs.get('search_radius', 1)  # 3x3max
        }
    elif intensity_method == 'raw':
        default_kwargs = {
            'search_radius': method_kwargs.get('search_radius', 0)
        }
    else:
        default_kwargs = {}
    
    final_method_kwargs = {**default_kwargs, **method_kwargs}
    
    # Collect tasks for parallel processing
    intensity_tasks = []
    
    for cyc in range(1, cyc_num + 1):
        for channel in channels:
            # For PRISM, stitched images are directly in stitched directory
            # Format: cyc_1_channel.tif
            if registered_dir.is_file():
                # If registered_dir is actually a file path, use it directly
                tile_path = registered_dir
            else:
                # Otherwise, look for cyc_n_channel.tif in the directory
                tile_path = registered_dir / f'cyc_{cyc}_{channel}.tif'
            
            if tile_path.exists():
                intensity_tasks.append((str(tile_path), channel, cyc, coords, intensity_method, final_method_kwargs))
            else:
                # Image doesn't exist, will be set to NaN
                intensity_df[f'cyc_{cyc}_{channel}'] = np.nan
    
    # Parallel intensity reading
    if intensity_tasks:
        n_workers = min(len(intensity_tasks), max_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_read_intensity_for_single_image, task) for task in intensity_tasks]
            for future in futures:
                cyc, channel, intensities, error = future.result()
                if error is None and intensities is not None:
                    intensity_df[f'cyc_{cyc}_{channel}'] = intensities
                else:
                    logger.warning(f"Failed to read intensity for cyc_{cyc}_{channel}: {error}")
                    intensity_df[f'cyc_{cyc}_{channel}'] = np.nan

    return intensity_df
