"""
Readout module for spot detection and intensity extraction.

This module provides:
- Spot detection using spotiflow or traditional methods
- Intensity extraction using tophat+3x3max or other methods
- Deduplication of detected spots
"""

from .spot_detection import (
    get_spot_coordinates,
    get_coordinates_for_tile,
    feature_tophat,
    feature_gaussian_tophat,
)

from .intensity_readout import (
    read_intensity_tophat,
    read_intensity_raw,
    get_intensity_df_for_tile,
)

from .deduplicate import deduplicate_coordinates, deduplicate_dataframe

__all__ = [
    # Spot detection
    'get_spot_coordinates',
    'get_coordinates_for_tile',
    'feature_tophat',
    'feature_gaussian_tophat',
    # Intensity reading
    'read_intensity_tophat',
    'read_intensity_raw',
    'get_intensity_df_for_tile',
    # Deduplication
    'deduplicate_coordinates',
    'deduplicate_dataframe',
]
