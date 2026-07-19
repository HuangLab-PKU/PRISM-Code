"""
Base classes and interfaces for signal classification framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Container for classification results."""

    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    centroids: Optional[np.ndarray] = None
    model_params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dataframe(
        self,
        original_data: pd.DataFrame,
        confidence_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Convert results to DataFrame with soft classification output.

        Columns produced:

        * ``top1_label``, ``top1_prob`` -- best-matching component index and
          its probability.
        * ``top2_label``, ``top2_prob`` -- second-best component.
        * ``top1_gene``, ``top2_gene`` -- gene names (only when a codebook is
          available in ``self.metadata``).
        * ``predicted_label``, ``prediction_confidence`` -- kept for backward
          compatibility with old GMMMethod consumers.
        * ``is_confident`` -- only when ``confidence_threshold`` is given: True
          where ``prediction_confidence >= confidence_threshold``. Spots without
          probabilities are treated as not confident.

        Args:
            original_data: unused positionally, kept for API compatibility.
            confidence_threshold: if set, add the ``is_confident`` QC column.
        """
        result_df = pd.DataFrame()

        if self.probabilities is not None:
            top2_idx = np.argsort(self.probabilities, axis=1)[:, -2:][:, ::-1]
            n = len(top2_idx)

            result_df["top1_label"] = top2_idx[:, 0]
            result_df["top1_prob"] = self.probabilities[np.arange(n), top2_idx[:, 0]]
            result_df["top2_label"] = top2_idx[:, 1]
            result_df["top2_prob"] = self.probabilities[np.arange(n), top2_idx[:, 1]]

            # Map integer labels to gene names when a codebook is available
            codebook = (self.metadata or {}).get("codebook")
            if codebook is not None and "gene" in codebook.columns:
                gene_names = codebook["gene"].values
                result_df["top1_gene"] = gene_names[top2_idx[:, 0]]
                result_df["top2_gene"] = gene_names[top2_idx[:, 1]]

            # Backward-compatible columns
            result_df["predicted_label"] = top2_idx[:, 0]
            result_df["prediction_confidence"] = result_df["top1_prob"]

            if confidence_threshold is not None:
                result_df["is_confident"] = (
                    result_df["prediction_confidence"] >= confidence_threshold
                )
        else:
            result_df["top1_label"] = self.labels
            result_df["top1_prob"] = np.nan
            result_df["top2_label"] = np.nan
            result_df["top2_prob"] = np.nan
            result_df["predicted_label"] = self.labels
            result_df["prediction_confidence"] = np.nan

            if confidence_threshold is not None:
                # No probabilities -> confidence undefined -> treat as not confident
                result_df["is_confident"] = False

        return result_df


class BaseClassifier(ABC):
    """Abstract base class for signal point classifiers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize classifier with configuration.

        Args:
            config: Configuration dictionary containing method-specific parameters
        """
        self.config = config
        self.is_fitted = False
        self.model = None

    @abstractmethod
    def fit(
        self, features: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> "BaseClassifier":
        """
        Fit the classifier to training data.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Optional ground truth labels for supervised learning

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> ClassificationResult:
        """
        Predict labels for new data.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            ClassificationResult containing predictions and metadata
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.

        Returns:
            Feature importance array or None if not available
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid
        """
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.

        Returns:
            Dictionary containing model information
        """
        return {
            "method": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "config": self.config,
        }


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from raw signal data.

        Args:
            data: DataFrame containing signal intensity data

        Returns:
            Feature matrix (n_samples, n_features)
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.

        Returns:
            List of feature names
        """
        pass


class BasePreprocessor(ABC):
    """Abstract base class for data preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw signal data.

        Args:
            data: Raw signal data

        Returns:
            Preprocessed data
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and content.

        Args:
            data: Input data to validate

        Returns:
            True if data is valid
        """
        # Check for either PRISM channel names or unified channel names (from spot detection)
        prism_channels = ["R", "Ye", "B", "G"]
        unified_channels = ["ch1", "ch2", "ch3", "ch4"]

        has_prism_channels = all(col in data.columns for col in prism_channels)
        has_unified_channels = all(col in data.columns for col in unified_channels)

        return has_prism_channels or has_unified_channels
