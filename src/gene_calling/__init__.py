"""
Gene Calling Module

This module provides a flexible framework for classifying signal points based on
multi-channel intensity values. The framework supports various classification methods
including GMM, and can be easily extended with new algorithms.

Key Features:
- Multi-channel signal point classification
- Support for different classification algorithms
- Configurable preprocessing and feature extraction
- Comprehensive evaluation and visualization
- Manual threshold adjustment capabilities
- Quantitative evaluation tools
- Integration with existing PRISM pipeline
"""

from .base import BaseClassifier, ClassificationResult
from .gmm_method import GMMMethod
from .manual_method import ManualThresholdAdjuster
from .evaluator import ClassificationEvaluator, QuantitativeEvaluator
from .pipeline import SignalClassificationPipeline
from .config_loader import load_gene_calling_config, validate_gene_calling_config

__version__ = "1.0.0"
__all__ = [
    "BaseClassifier",
    "ClassificationResult",
    "GMMMethod",
    "ManualThresholdAdjuster",
    "ClassificationEvaluator",
    "QuantitativeEvaluator",
    "SignalClassificationPipeline",
    "load_gene_calling_config",
    "validate_gene_calling_config",
]
