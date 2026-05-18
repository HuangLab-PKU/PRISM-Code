"""
Channel correction utilities for gene calling.

Provides a shared correction function used by both old GMMMethod and new
CodebookGMMMethod via the pipeline preprocess step.

The correction performs three operations in sequence:
1. Rename columns from excitation-wavelength names to fluorophore names
2. Apply a 4x4 linear unmixing matrix (spectral correction)
3. Clip negative values to zero
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_channel_correction(
    data: pd.DataFrame,
    channel_config: Dict[str, dict],
    transform_matrix: Union[np.ndarray, list],
    clip_negative: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Apply channel correction: rename + matrix transform + clip.

    Parameters
    ----------
    data : pd.DataFrame
        Raw intensity data. Must contain all columns listed as keys in
        *channel_config*.  Additional columns (e.g. ``index``, coordinates)
        are preserved.
    channel_config : dict
        Mapping from intensity.csv column name to metadata dict.  Each value
        must contain at least ``{"fluorophore": "<name>"}``.  Example::

            {
                "r01_ex628": {"fluorophore": "r01_Cy5",  "role": "color", ...},
                "r01_ex531": {"fluorophore": "r01_Cy3",  "role": "layer", ...},
                ...
            }

        The order of keys defines the column order for matrix multiplication.
    transform_matrix : array-like, shape (N, N)
        Unmixing matrix.  ``corrected = raw @ M.T`` where *raw* is the
        row-vector of intensities in the order given by *channel_config* keys.
        Use ``np.eye(N)`` (identity) for no correction.
    clip_negative : bool
        If True, clip all corrected values to >= 0.
    output_path : Path or str, optional
        If provided, save ``intensity_corrected.csv`` to this path.

    Returns
    -------
    pd.DataFrame
        A copy of *data* with excitation columns replaced by corrected
        fluorophore-named columns.
    """
    # Resolve column lists in config order
    excitation_cols: List[str] = list(channel_config.keys())
    fluorophore_cols: List[str] = [v["fluorophore"] for v in channel_config.values()]

    # Validate that all excitation columns exist in the data
    missing = [c for c in excitation_cols if c not in data.columns]
    if missing:
        raise KeyError(
            f"Intensity data is missing expected columns: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    M = np.asarray(transform_matrix, dtype=np.float64)
    n_channels = len(excitation_cols)
    if M.shape != (n_channels, n_channels):
        raise ValueError(
            f"transform_matrix shape {M.shape} does not match "
            f"number of channels ({n_channels})"
        )

    # Extract raw intensities in config order
    raw = data[excitation_cols].values.astype(np.float64)  # (N, C)

    # Apply linear correction: corrected = raw @ M^T
    corrected = raw @ M.T

    if clip_negative:
        np.maximum(corrected, 0, out=corrected)

    # Build result: drop excitation columns, add fluorophore columns
    result = data.drop(columns=excitation_cols, errors="ignore").copy()
    for i, col_name in enumerate(fluorophore_cols):
        result[col_name] = corrected[:, i]

    logger.info(
        "Channel correction applied: %s -> %s (matrix det=%.4f, clip=%s)",
        excitation_cols,
        fluorophore_cols,
        np.linalg.det(M),
        clip_negative,
    )

    # Optionally persist corrected data
    if output_path is not None:
        output_path = Path(output_path)
        save_cols = [c for c in ["index"] + fluorophore_cols if c in result.columns]
        result[save_cols].to_csv(output_path, index=False)
        logger.info("Saved intensity_corrected.csv to %s", output_path)

    return result
