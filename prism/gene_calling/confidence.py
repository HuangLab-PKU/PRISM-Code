"""Posterior-confidence utilities for gene-calling QC filtering.

The decode confidence of a spot is its top-1 GMM posterior probability. Spots whose
top-1 posterior falls below a threshold sit between codewords (ambiguous) and can be
filtered out to obtain a cleaner, higher-specificity gene-call set / colourspace.
"""
from typing import Optional

import numpy as np


def top1_confidence(probabilities: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Top-1 posterior confidence per spot (row-wise max of the probability matrix).

    Args:
        probabilities: (n_spots, n_classes) posterior matrix, or None.

    Returns:
        (n_spots,) array of max posterior per spot, or None if probabilities is None.
    """
    if probabilities is None:
        return None
    return np.asarray(probabilities, dtype=float).max(axis=1)


def confident_mask(probabilities: Optional[np.ndarray], threshold: float) -> np.ndarray:
    """Boolean mask of spots whose top-1 posterior is >= threshold.

    Spots with no probabilities available are treated as not confident.
    """
    conf = top1_confidence(probabilities)
    if conf is None:
        return np.zeros(0, dtype=bool)
    return conf >= threshold
