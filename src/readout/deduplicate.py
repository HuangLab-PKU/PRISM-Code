"""Deduplication module for removing duplicate spots."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import logging

logger = logging.getLogger('readout_deduplicate')


def deduplicate_coordinates(coordinates, threshold=2, scores=None):
    """Remove duplicate coordinates within a threshold distance.
    
    If multiple coordinates are within the threshold distance, keep the one
    with the highest score (if scores provided) or the first one.
    
    Parameters
    ----------
    coordinates : np.ndarray
        (N, 2) array of (Y, X) coordinates
    threshold : float
        Distance threshold for considering coordinates as duplicates (default: 2 pixels)
    scores : np.ndarray, optional
        (N,) array of scores for each coordinate. Higher scores are preferred.
        If None, first coordinate in each group is kept.
        
    Returns
    -------
    tuple
        (kept_indices, deduplicated_coordinates)
        kept_indices: array of indices of kept coordinates
        deduplicated_coordinates: (M, 2) array of deduplicated coordinates
    """
    if len(coordinates) == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=coordinates.dtype)
    
    coords = np.asarray(coordinates)
    
    # Build KDTree for efficient distance queries
    tree = cKDTree(coords)
    
    # Find pairs within threshold
    pairs = tree.query_pairs(r=threshold, p=float('inf'))  # Chebyshev distance (box region)
    
    if len(pairs) == 0:
        # No duplicates found
        return np.arange(len(coords)), coords
    
    # Find connected components (groups of duplicates)
    n_points = len(coords)
    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    # Create symmetric adjacency matrix
    data = np.ones(len(pairs), dtype=bool)
    adj = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    # connected_components returns (n_components, labels)
    n_components, labels = connected_components(adj, directed=False)
    
    # Select best coordinate per component
    if scores is not None:
        scores = np.asarray(scores)
        # Find index of max score per label
        kept_indices = []
        for label_id in range(n_components):
            mask = labels == label_id
            if np.any(mask):
                # Get indices where mask is True
                label_indices = np.where(mask)[0]
                # Find index with maximum score
                best_idx = label_indices[np.argmax(scores[label_indices])]
                kept_indices.append(best_idx)
    else:
        # Keep first coordinate in each component
        kept_indices = []
        for label_id in range(n_components):
            mask = labels == label_id
            if np.any(mask):
                # Get first index where mask is True
                first_idx = np.where(mask)[0][0]
                kept_indices.append(first_idx)
    
    kept_indices = np.array(kept_indices, dtype=int)
    deduplicated_coords = coords[kept_indices]
    
    return kept_indices, deduplicated_coords


def deduplicate_dataframe(df, coordinate_columns=['Y', 'X'], threshold=2, sort_by=None, ascending=False):
    """Remove duplicate rows based on coordinate proximity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing coordinate columns and other data
    coordinate_columns : list
        List of column names for coordinates (default: ['Y', 'X'])
    threshold : float
        Distance threshold for considering coordinates as duplicates (default: 2 pixels)
    sort_by : str, optional
        Column name to sort by before deduplication. Higher values are preferred.
        If None, first row in each group is kept.
    ascending : bool
        If True, sort ascending (lower values preferred). If False, sort descending (higher values preferred).
        
    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame
    """
    if len(df) == 0:
        return df.copy()
    
    # Sort by sort_by column if provided
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, inplace=False)
    
    # Extract coordinates
    coordinates = df[coordinate_columns].values
    
    # Calculate scores if sort_by provided
    scores = None
    if sort_by is not None and sort_by in df.columns:
        scores = df[sort_by].values
    
    # Deduplicate
    kept_indices, _ = deduplicate_coordinates(coordinates, threshold=threshold, scores=scores)
    
    # Return deduplicated dataframe
    df_dedup = df.iloc[kept_indices].copy()
    
    logger.info(f'Deduplication: {len(df)} -> {len(df_dedup)} spots (removed {len(df) - len(df_dedup)})')
    
    return df_dedup
