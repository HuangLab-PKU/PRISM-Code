"""
Data processing and feature extraction for signal classification.
"""

import pandas as pd
import numpy as np
import cv2
import os
import re
from typing import Dict, Any, List
from tqdm import tqdm
from .base import BasePreprocessor, BaseFeatureExtractor


class SignalDataProcessor(BasePreprocessor):
    """
    Preprocessor for signal point data following PRISM methodology.

    Handles:
    - Crosstalk elimination
    - Intensity scaling
    - Normalization
    - Gaussian blur for boundary points
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Gene calling: no scaling/crosstalk/FRET adjustments here (handled in spot detection)
        self.scaling_factors = None
        self.crosstalk_factor = None
        self.fret_adjustments = None
        self.gaussian_noise_scale = config.get("gaussian_noise_scale", 0.01)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing pipeline to signal data.

        Note: Crosstalk elimination and intensity scaling are already done in spot detection.
        This method focuses on ratio calculations and transformations for gene calling.

        Args:
            data: Intensity data with columns ['ch1', 'ch2', 'ch3', 'ch4'] (from spot detection)

        Returns:
            Preprocessed data with additional computed columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required channels")

        processed = data.copy()

        # Use standard chn naming directly (no PRISM renaming)

        # Step 1: Calculate sum and ratios (skip crosstalk and scaling - already done)
        processed = self._calculate_ratios(processed)

        # Normalize ratios to sum to 1
        processed = self._normalize_ratios(processed)

        # Apply Gaussian blur for boundary points
        processed = self._apply_gaussian_blur(processed)

        # Calculate projection coordinates
        processed = self._calculate_projections(processed)

        return processed

    def _eliminate_crosstalk(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _scale_intensities(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _calculate_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate channel ratios and sum."""
        data["sum"] = data["ch1"] + data["ch2"] + data["ch4"]

        # Calculate ratios
        data["ch1/A"] = data["ch1"] / data["sum"]
        data["ch2/A"] = data["ch2"] / data["sum"]
        data["ch3/A"] = data["ch3"] / data["sum"]
        data["ch4/A"] = data["ch4"] / data["sum"]

        # Cap G/A ratio
        data.loc[data["ch3/A"] > 5, "ch3/A"] = 5

        return data

    def _apply_fret_adjustments(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _transform_g_channel(self, data: pd.DataFrame) -> pd.DataFrame:
        """No G-channel transforms in gene calling."""
        return data

    def _normalize_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize color ratios to sum to 1 (ch1,ch2,ch4)."""
        data["sum_scale"] = data["ch1/A"] + data["ch2/A"] + data["ch4/A"]
        data["ch1/A"] = data["ch1/A"] / data["sum_scale"]
        data["ch2/A"] = data["ch2/A"] / data["sum_scale"]
        data["ch4/A"] = data["ch4/A"] / data["sum_scale"]
        return data

    def _apply_gaussian_blur(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Gaussian noise to boundary points."""
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]

        # Add noise to zero values
        for col in ratio_cols:
            zero_mask = data[col] == 0
            if zero_mask.any():
                noise = np.random.normal(0, self.gaussian_noise_scale, zero_mask.sum())
                data.loc[zero_mask, col] = noise

        # Add noise to boundary values (1.0)
        for col in ratio_cols:
            boundary_mask = data[col] == 1
            if boundary_mask.any():
                noise = np.random.normal(
                    0, self.gaussian_noise_scale, boundary_mask.sum()
                )
                data.loc[boundary_mask, col] = 1 + noise

        return data

    def _calculate_projections(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate 2D projections for visualization."""
        # Unified projection vectors for ratio order [ch1/A, ch2/A, ch4/A]
        # Reordered from legacy ([ch2, ch4, ch1]) to maintain same geometry
        color_dim_1_proj = np.array([[0.0], [-np.sqrt(2) / 2], [np.sqrt(2) / 2]])
        color_dim_2_proj = np.array([[np.sqrt(2) / 2], [-1 / 2], [-1 / 2]])

        # Calculate projections
        ratio_matrix = data[["ch1/A", "ch2/A", "ch4/A"]].values
        data["color_dim_1"] = ratio_matrix @ color_dim_1_proj.flatten()
        data["color_dim_2"] = ratio_matrix @ color_dim_2_proj.flatten()

        return data


class SignalFeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor for signal classification.

    Extracts features based on the signal encoding scheme:
    - Three channel ratios (ch1/sum, ch2/sum, ch4/sum)
    - Fourth channel relative intensity (ch3/sum)
    - Additional derived features
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_types = config.get(
            "feature_types",
            ["ratios", "projections", "intensity_features", "statistical_features"],
        )
        self.include_g_channel = config.get("include_g_channel", True)

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from preprocessed signal data.

        Args:
            data: Preprocessed data from SignalDataProcessor

        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []

        if "ratios" in self.feature_types:
            features.append(self._extract_ratio_features(data))

        if "projections" in self.feature_types:
            features.append(self._extract_projection_features(data))

        if "intensity_features" in self.feature_types:
            features.append(self._extract_intensity_features(data))

        if "statistical_features" in self.feature_types:
            features.append(self._extract_statistical_features(data))

        return (
            np.column_stack(features)
            if features
            else np.array([]).reshape(len(data), 0)
        )

    def _extract_ratio_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract channel ratio features."""
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
        if self.include_g_channel:
            ratio_cols.append("ch3/A")
        return data[ratio_cols].values

    def _extract_projection_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract 2D projection features."""
        return data[["color_dim_1", "color_dim_2"]].values

    def _extract_intensity_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract intensity-based features."""
        features = []

        # Sum intensity
        features.append(data["sum"].values.reshape(-1, 1))

        # Log-transformed sum
        features.append(np.log1p(data["sum"]).values.reshape(-1, 1))

        # Individual channel intensities
        for channel in ["ch1", "ch2", "ch3", "ch4"]:
            if channel in data.columns:
                features.append(data[channel].values.reshape(-1, 1))

        return (
            np.column_stack(features)
            if features
            else np.array([]).reshape(len(data), 0)
        )

    def _extract_statistical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract statistical features."""
        features = []

        # Ratio statistics
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
        for col in ratio_cols:
            features.append(data[col].values.reshape(-1, 1))

        # Cross-ratios
        features.append((data["ch1/A"] / (data["ch2/A"] + 1e-8)).values.reshape(-1, 1))
        features.append((data["ch4/A"] / (data["ch1/A"] + 1e-8)).values.reshape(-1, 1))
        features.append((data["ch2/A"] / (data["ch4/A"] + 1e-8)).values.reshape(-1, 1))

        return (
            np.column_stack(features)
            if features
            else np.array([]).reshape(len(data), 0)
        )

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        names = []

        if "ratios" in self.feature_types:
            ratio_names = ["ch1/A", "ch2/A", "ch4/A"]
            if self.include_g_channel:
                ratio_names.append("ch3/A")
            names.extend(ratio_names)

        if "projections" in self.feature_types:
            names.extend(["color_dim_1", "color_dim_2"])

        if "intensity_features" in self.feature_types:
            names.extend(["sum", "log_sum"])
            for channel in ["ch1", "ch2", "ch3", "ch4"]:
                names.append(f"{channel}")

        if "statistical_features" in self.feature_types:
            names.extend(["ch1/A", "ch2/A", "ch4/A"])
            names.extend(["ch1/ch2_ratio", "ch4/ch1_ratio", "ch2/ch4_ratio"])

        return names


## ManualThresholdAdjuster removed from this module; use gene_calling.manual_method instead
