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

    def to_dataframe(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Convert results to DataFrame with only label and confidence."""
        # Create a minimal DataFrame with only prediction results
        result_df = pd.DataFrame({"predicted_label": self.labels})

        if self.probabilities is not None:
            result_df["prediction_confidence"] = np.max(self.probabilities, axis=1)

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
