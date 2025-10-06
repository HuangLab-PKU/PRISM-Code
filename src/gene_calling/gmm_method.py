"""
GMM-based gene calling method implementation.

This module combines GMM-specific preprocessing and classification into a unified method.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.mixture import GaussianMixture
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import logging
import time

from .base import ClassificationResult

logger = logging.getLogger(__name__)


class GMMMethod:
    """
    Complete GMM-based gene calling method.

    Combines GMM-specific preprocessing and classification into a unified approach.
    This method handles the entire pipeline from raw intensity data to classification results.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GMM method with configuration.

        Args:
            config: Configuration dictionary containing GMM-specific parameters
        """
        self.config = config
        self.is_fitted = False
        self.model = None
        self.training_data = None
        self.training_features = None

        # GMM-specific preprocessing parameters
        self.gaussian_noise_scale = config.get("gaussian_noise_scale", 0.01)
        self.color_grade = config.get("color_grade", 5)  # For backward compatibility
        self.layer_grade = config.get("layer_grade", 2)

        # PRISM panel parameters (from base config)
        self.prism_panel_type = config.get("prism_panel_type", "PRISM30")
        self.num_per_layer = config.get("num_per_layer", 15)
        self.g_layer_num = config.get("g_layer_num", 2)
        self.total_components = config.get("total_components", 30)

        # Centroid movement constraint (post-fit enforcement)
        self.centroid_max_movement = config.get("centroid_max_movement", 0.15)
        self.constrained_fit_rounds = config.get("constrained_fit_rounds", 3)

        # GMM classification parameters
        self.gmm_config = config.get("gmm", {})
        self.covariance_type = self.gmm_config.get("covariance_type", "diag")
        self.use_layers = self.gmm_config.get("use_layers", True)
        # Option to skip layering and cluster 30 components directly using 4D ratios
        self.use_4d_ratios_only = self.gmm_config.get("use_4d_ratios_only", False)
        # Include special class (only layer chn bright) into GMM instead of post-hoc label
        self.include_special_in_gmm = self.gmm_config.get("include_special_in_gmm", True)

        # Feature extraction parameters
        self.feature_types = config.get(
            "feature_types",
            ["ratios", "projections", "intensity_features", "statistical_features"],
        )
        self.include_g_channel = config.get("include_g_channel", True)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply GMM-specific preprocessing to signal data.

        Note: Crosstalk elimination and intensity scaling are already done in spot detection.
        This method focuses on GMM-specific transformations.

        Args:
            data: Intensity data with columns ['ch1', 'ch2', 'ch3', 'ch4'] (from spot detection)

        Returns:
            Preprocessed data with additional computed columns
        """
        if not self._validate_data(data):
            raise ValueError("Input data missing required channels")

        processed = data.copy()

        # Ensure we have standard ch1-ch4 channel names (from spot detection)
        # If data already has ch1-ch4, use them directly
        # Ensure data has standard ch1-ch4 naming
        required_channels = ["ch1", "ch2", "ch3", "ch4"]
        missing_channels = [
            col for col in required_channels if col not in processed.columns
        ]
        if missing_channels:
            raise ValueError(f"Missing required channels: {missing_channels}")

        # Step 1: Calculate sum and ratios (skip crosstalk and scaling - already done)
        processed = self._calculate_ratios(processed)

        # Step 2: Apply Gaussian blur for boundary points
        processed = self._apply_gaussian_blur(processed)

        # Step 3: Calculate projection coordinates
        processed = self._calculate_projections(processed)

        # Step 4: Assign G layers based on ch4/A distribution
        processed = self._assign_g_layers(processed)

        return processed

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from preprocessed signal data.

        Args:
            data: Preprocessed data from GMM preprocessing

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

    def fit(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> "GMMMethod":
        """
        Fit the GMM method to training data.

        Args:
            data: Training data with intensity values
            labels: Optional ground truth labels (not used in unsupervised GMM)
            metadata: Optional metadata (e.g., G_layer information)

        Returns:
            Self for method chaining
        """
        logger.info("Fitting GMM method...")

        # Store training data
        self.training_data = data.copy()

        # Preprocess data
        preprocess_start = time.time()
        processed_data = self.preprocess(data)
        preprocess_time = time.time() - preprocess_start
        logger.info(f"Preprocessing took: {preprocess_time:.2f} seconds")

        # Extract features
        feature_start = time.time()
        features = self.extract_features(processed_data)
        feature_time = time.time() - feature_start

        # Clean features: remove any NaN or inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        logger.info(
            f"Cleaned features: shape={features.shape}, any NaN: {np.isnan(features).any()}, any inf: {np.isinf(features).any()}"
        )
        logger.info(f"Feature extraction took: {feature_time:.2f} seconds")
        self.training_features = features

        # Prepare G-layer from processed data to ensure both layers are trained
        if self.use_layers:
            if isinstance(processed_data, pd.DataFrame) and "G_layer" in processed_data.columns:
                g_layer = processed_data["G_layer"].to_numpy()
            else:
                g_layer = np.zeros(len(processed_data))
        else:
            g_layer = np.zeros(len(processed_data))

        # Fit GMM model
        logger.info(
            f"Starting GMM fitting: use_layers={self.use_layers}, g_layer_num={self.g_layer_num}, num_per_layer={self.num_per_layer}"
        )
        if self.use_layers and not self.use_4d_ratios_only:
            logger.info("About to call _fit_layered_gmm")
            self.model = self._fit_layered_gmm(features, g_layer, processed_data)
            logger.info(
                f"_fit_layered_gmm completed, model keys: {list(self.model.keys()) if isinstance(self.model, dict) else 'not dict'}"
            )
        else:
            # Single 30-component GMM on 4D ratios (using ch1/ch2/ch4 and layer ch3)
            logger.info("About to call _fit_single_gmm (4D ratios)")
            ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
            if self.include_g_channel:
                ratio_cols.append("ch3/A")
            ratio_matrix = processed_data[ratio_cols].values
            # Append a synthetic special-class centroid near [0,0,0, high] by augmenting training set with a small cluster if desired
            # We avoid data duplication; instead, we rely on initial kmeans to discover it. If needed we can seed means later.
            ratio_matrix = np.nan_to_num(ratio_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            self.model = self._fit_single_gmm(ratio_matrix)
            logger.info("_fit_single_gmm completed")

        self.is_fitted = True
        logger.info("GMM method fitted successfully")

        return self

    def predict(self, data: pd.DataFrame) -> ClassificationResult:
        """
        Predict labels for new data.

        Args:
            data: Data to predict (can be same as training data)

        Returns:
            ClassificationResult containing predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before making predictions")

        logger.info("Making predictions with GMM method...")

        # Preprocess data
        processed_data = self.preprocess(data)

        # Handle special case: sum=0 but G>0 (ch3>0)
        # If using 4D single-GMM and include_special_in_gmm=True, we do NOT peel them off;
        # they will naturally form a cluster near [0,0,0, cap(ch3/A)]. Otherwise, assign special label.
        special_mask = (processed_data["sum"] < 1.0) & (processed_data["ch3"] > 0)
        n_special = special_mask.sum()

        if n_special > 0 and not (self.use_4d_ratios_only and self.include_special_in_gmm):
            logger.info(
                f"Found {n_special} points with sum=0 but ch3>0, assigning to PRISMpanel + 1"
            )
            # Calculate total number of PRISM panels (assuming 15 per layer, 2 layers = 30 total)
            total_prism_panels = self.num_per_layer * self.g_layer_num
            special_label = total_prism_panels + 1  # PRISMpanel + 1

            # Create labels array with special assignment
            labels = np.full(len(processed_data), -1, dtype=int)  # Initialize with -1
            labels[special_mask] = special_label

            # Create probabilities array (1.0 confidence for special points)
            # Need space for total_prism_panels (30) + special_label (31) = 32 total
            probabilities = np.zeros((len(processed_data), total_prism_panels + 2))
            probabilities[special_mask, special_label] = 1.0

            # For non-special points, use normal GMM prediction
            normal_mask = ~special_mask
            if normal_mask.any():
                normal_features = self.extract_features(processed_data[normal_mask])

                if self.use_layers and not self.use_4d_ratios_only:
                    logger.info(
                        f"Using layered GMM prediction with {len(self.model)} layers for {normal_mask.sum()} normal points"
                    )
                    normal_labels, normal_probs, centroids = self._predict_layered_gmm(
                        normal_features, processed_data[normal_mask]
                    )
                else:
                    logger.info("Using single GMM prediction for normal points (4D ratios)")
                    ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
                    if self.include_g_channel:
                        ratio_cols.append("ch3/A")
                    ratio_matrix = processed_data.loc[normal_mask, ratio_cols].values
                    ratio_matrix = np.nan_to_num(ratio_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
                    normal_labels, normal_probs, centroids = self._predict_single_gmm(
                        ratio_matrix
                    )

                # Assign normal predictions
                labels[normal_mask] = normal_labels
                probabilities[normal_mask, :total_prism_panels] = normal_probs
            else:
                centroids = {}
        else:
            # Use normal prediction (special points included into GMM if in 4D mode)
            if self.use_layers and not self.use_4d_ratios_only:
                features = self.extract_features(processed_data)
                logger.info(
                    f"Using layered GMM prediction with {len(self.model)} layers"
                )
                labels, probabilities, centroids = self._predict_layered_gmm(
                    features, processed_data
                )
            else:
                logger.info("Using single GMM prediction (4D ratios)")
                ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
                if self.include_g_channel:
                    ratio_cols.append("ch3/A")
                ratio_matrix = processed_data[ratio_cols].values
                ratio_matrix = np.nan_to_num(ratio_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
                labels, probabilities, centroids = self._predict_single_gmm(ratio_matrix)

        # Ensure labels start from 1 (global) and handle any unlabeled (-1)
        if labels.min() <= 0:
            # If labels appear 0-based, shift to 1-based
            labels = labels + 1
        unlabeled = labels < 1
        if np.any(unlabeled):
            # Fill unlabeled by global argmax probability index + 1
            labels[unlabeled] = np.argmax(probabilities[unlabeled], axis=1) + 1

        # Create result
        result = ClassificationResult(
            labels=labels,
            probabilities=probabilities,
            centroids=centroids,
            model_params=self.get_model_info(),
            metadata={"method": "GMM", "use_layers": self.use_layers},
        )

        logger.info(f"Predicted {len(np.unique(labels))} clusters")

        return result

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required channels."""
        # Check for standard ch1-ch4 channel names (from spot detection)
        standard_channels = ["ch1", "ch2", "ch3", "ch4"]
        has_standard_channels = all(col in data.columns for col in standard_channels)

        # Also support legacy PRISM channel names for backward compatibility
        prism_channels = ["R", "Ye", "B", "G"]
        has_prism_channels = all(col in data.columns for col in prism_channels)

        return has_standard_channels or has_prism_channels

    def _calculate_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate channel ratios and sum using standard ch1-ch4 naming."""
        # Use standard ch1-ch4 naming system (like spot detection)
        # ch1: Cy5 (670nm), ch2: TxRed (615nm), ch3: Cy3 (550nm), ch4: FAM (520nm)
        # ch3 is the layering channel (G), ch1/ch2/ch4 are the color channels

        # Calculate sum using only color channels (excluding layering channel ch3)
        data["sum"] = data["ch1"] + data["ch2"] + data["ch4"]

        # Calculate ratios using standard ch1-ch4 naming
        data["ch1/A"] = data["ch1"] / data["sum"]
        data["ch2/A"] = data["ch2"] / data["sum"]
        data["ch3/A"] = data["ch3"] / data["sum"]  # Layering channel ratio
        data["ch4/A"] = data["ch4"] / data["sum"]

        # Cap ch3/A ratio (layering channel) to reasonable range
        data.loc[data["ch3/A"] > 5, "ch3/A"] = 5

        return data

    def _apply_gaussian_blur(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Gaussian noise to boundary points."""
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]  # Color channels only

        # Add noise to zero values (color channels)
        for col in ratio_cols:
            zero_mask = data[col] == 0
            if zero_mask.any():
                noise = np.random.normal(0, self.gaussian_noise_scale, zero_mask.sum())
                data.loc[zero_mask, col] = noise

        # Add noise to boundary values (1.0) (color channels)
        for col in ratio_cols:
            boundary_mask = data[col] == 1
            if boundary_mask.any():
                noise = np.random.normal(
                    0, self.gaussian_noise_scale, boundary_mask.sum()
                )
                data.loc[boundary_mask, col] = 1 + noise

        # Add slight noise at zero for layering channel ch3/A (no boundary-1 noise)
        if "ch3/A" in data.columns:
            g_zero_mask = data["ch3/A"] == 0
            if g_zero_mask.any():
                noise = np.random.normal(0, self.gaussian_noise_scale * 0.5, g_zero_mask.sum())
                data.loc[g_zero_mask, "ch3/A"] = noise
            # ensure bounds after noise
            data["ch3/A"] = np.clip(data["ch3/A"].astype(float), 0.0, 5.0)

        return data

    def _calculate_projections(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate 2D projections for visualization."""
        # Define projection vectors (legacy geometry) but using unified order [ch1/A, ch2/A, ch4/A]
        # Legacy order was [ch2/A, ch4/A, ch1/A] with:
        #   color_dim_1_proj_old = [-sqrt(2)/2, +sqrt(2)/2, 0]
        #   color_dim_2_proj_old = [-1/2, -1/2, +sqrt(2)/2]
        # Reorder to new order [ch1, ch2, ch4] i.e., indices [2, 0, 1] from old:
        #   color_dim_1_proj_new = [0, -sqrt(2)/2, +sqrt(2)/2]
        #   color_dim_2_proj_new = [ +sqrt(2)/2, -1/2, -1/2 ]
        color_dim_1_proj = np.array([[0.0], [-np.sqrt(2) / 2], [np.sqrt(2) / 2]])
        color_dim_2_proj = np.array([[np.sqrt(2) / 2], [-1 / 2], [-1 / 2]])

        # Calculate projections using unified channel order
        Q_CHNS = ["ch1/A", "ch2/A", "ch4/A"]

        # Ensure all required ratio columns exist
        if not all(col in data.columns for col in Q_CHNS):
            raise ValueError(f"Missing required ratio columns: {Q_CHNS}")

        # Calculate color dimension coordinates using matrix multiplication
        data["color_dim_1"] = data[Q_CHNS] @ color_dim_1_proj.flatten()
        data["color_dim_2"] = data[Q_CHNS] @ color_dim_2_proj.flatten()

        return data

    def _assign_g_layers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign G layers based on ch3/A distribution (layering channel)."""
        if self.g_layer_num <= 1:
            data["G_layer"] = 0
            return data

        # Find ch3/A minima for layer boundaries (similar to legacy code)
        from scipy import stats
        from scipy.signal import find_peaks

        # Get ch3/A values for layering (only non-zero values for better KDE)
        ch3_a_values = data[data["ch3/A"] > 0]["ch3/A"].values

        if len(ch3_a_values) == 0:
            # If no non-zero values, assign all to layer 0
            data["G_layer"] = 0
            logger.info("No non-zero ch3/A values found, assigning all to layer 0")
            return data

        # Use kernel density estimation to find minima
        kde = stats.gaussian_kde(
            ch3_a_values, bw_method=0.5
        )  # Smaller bandwidth for better resolution
        x_range = np.linspace(
            ch3_a_values.min(), ch3_a_values.max(), 2000
        )  # Higher resolution
        kde_values = kde(x_range)

        # Find local minima with better parameters
        minima_indices, _ = find_peaks(-kde_values, distance=100, prominence=0.01)
        minima = x_range[minima_indices]

        # If we find minima, use them to assign layers
        if len(minima) > 0:
            # Take only the first (g_layer_num - 1) minima
            minima = minima[: self.g_layer_num - 1]

            # Create bin boundaries
            bin_edges = (
                [ch3_a_values.min() - 0.01] + list(minima) + [ch3_a_values.max() + 0.01]
            )

            # Assign layers using pd.cut
            data["G_layer"] = pd.cut(
                data["ch3/A"],
                bins=bin_edges,
                labels=list(range(len(bin_edges) - 1)),
                include_lowest=True,
                right=False,
            ).astype(int)
        else:
            # If no minima found, try a simpler approach
            # Use percentile-based splitting
            non_zero_values = data[data["ch3/A"] > 0]["ch3/A"]
            if len(non_zero_values) > 0:
                median_val = non_zero_values.median()
                data["G_layer"] = 0  # Default to layer 0
                data.loc[data["ch3/A"] > median_val, "G_layer"] = 1
            else:
                data["G_layer"] = 0

        logger.info(
            f"Assigned G layers: {data['G_layer'].value_counts().sort_index().to_dict()}"
        )

        return data

    def _extract_ratio_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract channel ratio features."""
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]  # Color channels
        if self.include_g_channel:
            ratio_cols.append("ch3/A")  # Layering channel
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

        # Ratio statistics (color channels)
        ratio_cols = ["ch1/A", "ch2/A", "ch4/A"]
        for col in ratio_cols:
            features.append(data[col].values.reshape(-1, 1))

        # Cross-ratios (color channels)
        features.append((data["ch1/A"] / (data["ch2/A"] + 1e-8)).values.reshape(-1, 1))
        features.append((data["ch4/A"] / (data["ch1/A"] + 1e-8)).values.reshape(-1, 1))
        features.append((data["ch2/A"] / (data["ch4/A"] + 1e-8)).values.reshape(-1, 1))

        return (
            np.column_stack(features)
            if features
            else np.array([]).reshape(len(data), 0)
        )

    def _fit_layered_gmm(
        self, features: np.ndarray, g_layer: np.ndarray, processed_data: pd.DataFrame
    ) -> Dict[str, GaussianMixture]:
        """Fit layered GMM model with initialized centroids."""
        models = {}
        for layer in range(self.g_layer_num):
            layer_mask = g_layer == layer
            if np.sum(layer_mask) > 0:
                layer_features = features[layer_mask]
                layer_data = processed_data[layer_mask]

                # Determine number of components for this layer
                n_components = self.num_per_layer
                logger.debug(
                    f"Layer {layer}: fitting {n_components} components with {len(layer_features)} samples"
                )

                # Calculate initial centroids for this layer
                means_init = self._calculate_initial_centroids(layer_data, n_components)
                logger.debug(
                    f"Layer {layer}: calculated initial centroids shape {means_init.shape}"
                )

                # Fit GMM for this layer with initialized centroids (optimized parameters)
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",  # Use diagonal covariance like legacy
                    means_init=means_init,
                    random_state=42,
                    max_iter=100,  # allow more iters
                    tol=1e-3,  # stricter tol
                    warm_start=True,  # Use warm start for better convergence
                )
                gmm.fit(layer_features)
                # Enforce centroid movement constraint: project back if moved too far
                if hasattr(gmm, "means_") and means_init is not None:
                    diffs = gmm.means_ - means_init
                    norms = np.linalg.norm(diffs, axis=1)
                    exceed = norms > self.centroid_max_movement
                    if np.any(exceed):
                        adjusted = means_init + diffs * (self.centroid_max_movement / (norms + 1e-9))[:, None]
                        gmm.means_ = adjusted
                models[layer] = gmm

                logger.debug(
                    f"Layer {layer}: GMM fitted, converged={gmm.converged_}, n_iter={gmm.n_iter_}"
                )
                logger.info(
                    f"Fitted GMM for layer {layer}: {n_components} components, {len(layer_features)} samples"
                )
                logger.info(f"Layer {layer} GMM converged: {gmm.converged_}")
                logger.info(f"Layer {layer} GMM n_iter: {gmm.n_iter_}")
                # Check if all points are assigned to the same cluster
                layer_pred = gmm.predict(layer_features)
                unique_pred = np.unique(layer_pred)
                logger.debug(
                    f"Layer {layer}: unique predictions: {len(unique_pred)} clusters: {unique_pred}"
                )
                logger.info(
                    f"Layer {layer} unique predictions: {len(unique_pred)} clusters: {unique_pred}"
                )

        logger.debug(f"_fit_layered_gmm returning {len(models)} models")
        return models

    def _fit_single_gmm(self, features: np.ndarray) -> GaussianMixture:
        """Fit single GMM model."""
        n_components = self.num_per_layer * self.g_layer_num

        means_init = None
        if self.use_4d_ratios_only:
            # Build deterministic means_init similar to layered case: color plane (15) × ch3 layers (layer_grade)
            # features expected shape: (N, 4) with order [ch1/A, ch2/A, ch4/A, ch3/A]

            # Find maxima for ch1/ch2/ch4 by KDE (use existing helper via pandas Series-like arrays)
            color_grade = int(getattr(self, "color_grade", 5))
            layer_grade = int(getattr(self, "layer_grade", 2))

            ch1_vals = features[:, 0]
            ch2_vals = features[:, 1]
            ch4_vals = features[:, 2]
            ch3_vals = features[:, 3]

            try:
                ch1_maxima = self._find_histogram_maxima(pd.Series(ch1_vals), expected_count=color_grade, bw_adjust=0.15)
                ch2_maxima = self._find_histogram_maxima(pd.Series(ch2_vals), expected_count=color_grade, bw_adjust=0.15)
                ch4_maxima = self._find_histogram_maxima(pd.Series(ch4_vals), expected_count=color_grade, bw_adjust=0.15)
            except Exception:
                # Fallback to quantiles if KDE fails
                q = np.linspace(0.05, 0.95, color_grade)
                ch1_maxima = np.quantile(ch1_vals, q)
                ch2_maxima = np.quantile(ch2_vals, q)
                ch4_maxima = np.quantile(ch4_vals, q)

            if len(ch1_maxima) < color_grade:
                ch1_maxima = np.linspace(ch1_vals.min(), ch1_vals.max(), color_grade)
            if len(ch2_maxima) < color_grade:
                ch2_maxima = np.linspace(ch2_vals.min(), ch2_vals.max(), color_grade)
            if len(ch4_maxima) < color_grade:
                ch4_maxima = np.linspace(ch4_vals.min(), ch4_vals.max(), color_grade)

            # ring-spiral combinations for color plane (15 combos when color_grade=5)
            triplets = self._ring_spiral_order_unified(color_grade)
            color_means = np.array([
                [
                    ch1_maxima[i1],
                    ch2_maxima[i2],
                    ch4_maxima[i4],
                ]
                for (i1, i2, i4) in triplets
            ])

            # ch3 layer maxima (usually near 0 and ~0.24)
            try:
                ch3_maxima = self._find_histogram_maxima(pd.Series(ch3_vals), expected_count=layer_grade, bw_adjust=0.1)
            except Exception:
                ch3_maxima = np.array([0.0, 0.24])[:layer_grade] if layer_grade >= 2 else np.array([0.0])
            # bound to [0,0.35] and keep peaks near 0.24
            ch3_maxima = np.array(ch3_maxima, dtype=float)
            ch3_maxima = np.clip(ch3_maxima, 0.0, 0.35)
            if len(ch3_maxima) < layer_grade or not np.all(np.isfinite(ch3_maxima)):
                base = [0.0, 0.24]
                ch3_maxima = np.array(base[:layer_grade])
            else:
                # Enforce deterministic peaks near 0 and 0.3 for layer channel
                if layer_grade >= 2:
                    ch3_maxima = np.array([0.0, 0.3])

            # Tile color means across ch3 layers to reach total_components (e.g., 15*2=30)
            means_4d = []
            for v in ch3_maxima:
                tmp = np.hstack([color_means, np.full((color_means.shape[0], 1), v)])
                means_4d.append(tmp)
            means_4d = np.vstack(means_4d)

            # Limit to exactly total_components
            if means_4d.shape[0] > self.total_components:
                means_4d = means_4d[: self.total_components]
            means_init = means_4d
            n_components = means_init.shape[0]

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=42,
            max_iter=100,
            tol=1e-3,
            warm_start=True,
            init_params="kmeans" if means_init is None else "random",
            means_init=means_init,
        )
        gmm.fit(features)

        logger.info(
            f"Fitted single GMM: {n_components} components, {len(features)} samples"
        )

        return gmm

    def _calculate_initial_centroids(
        self, layer_data: pd.DataFrame, n_components: int
    ) -> np.ndarray:
        """Calculate initial centroids based on histogram maxima (from legacy code)."""

        logger.info(
            f"Calculating initial centroids for {n_components} components with {len(layer_data)} data points"
        )

        # Sample data for histogram analysis (similar to legacy code)
        # For large datasets, use a larger sample size
        sample_size = (
            min(500000, len(layer_data))
            if len(layer_data) > 100000
            else min(100000, len(layer_data))
        )
        sample_data = layer_data.sample(sample_size)
        logger.info(
            f"Sampled {len(sample_data)} data points for histogram analysis (from {len(layer_data)} total)"
        )

        # Calculate histogram maxima for each channel ratio (color channels)
        ch1_maxima = self._find_histogram_maxima(
            sample_data["ch1/A"], self.color_grade, bw_adjust=0.15
        )
        ch2_maxima = self._find_histogram_maxima(
            sample_data["ch2/A"], self.color_grade, bw_adjust=0.15
        )
        ch4_maxima = self._find_histogram_maxima(
            sample_data["ch4/A"], self.color_grade, bw_adjust=0.15
        )

        # Generate ring-spiral order directly in unified axes (i1,i2,i4) = (ch1,ch2,ch4)
        spiral_triplets = self._ring_spiral_order_unified(self.color_grade)

        ratio_centroids = np.array([
            [
                ch1_maxima[i1],  # ch1
                ch2_maxima[i2],  # ch2
                ch4_maxima[i4],  # ch4
            ]
            for (i1, i2, i4) in spiral_triplets
        ])

        # Ensure we have the right number of centroids
        if len(ratio_centroids) > n_components:
            ratio_centroids = ratio_centroids[:n_components]
        elif len(ratio_centroids) < n_components:
            # If we don't have enough centroids, duplicate the last one
            while len(ratio_centroids) < n_components:
                ratio_centroids = np.vstack([ratio_centroids, ratio_centroids[-1:]])

        # Now we need to expand these centroids to match the full feature space
        # Get the full feature space by extracting features from sample data
        sample_features = self.extract_features(sample_data)
        # Clean sample features to avoid NaN/Inf propagating to means_init
        sample_features = np.nan_to_num(
            sample_features, nan=0.0, posinf=1e6, neginf=-1e6
        )
        feature_dim = sample_features.shape[1]

        # Create full-dimensional centroids
        full_centroids = np.zeros((len(ratio_centroids), feature_dim))

        # Fill in the ratio features (first 3 dimensions)
        full_centroids[:, :3] = ratio_centroids

        # For other features, use the mean values from the sample data
        for i in range(3, feature_dim):
            col_mean = float(np.mean(sample_features[:, i]))
            if not np.isfinite(col_mean):
                col_mean = 0.0
            full_centroids[:, i] = col_mean

        # Ensure no NaN/Inf in centroids before passing to sklearn
        full_centroids = np.nan_to_num(full_centroids, nan=0.0, posinf=1e6, neginf=-1e6)

        # Apply centroid constraints to prevent centroids from moving too far from initial positions
        # This ensures that the 30 clusters don't exceed init + some range
        full_centroids = self._apply_centroid_constraints(
            full_centroids, sample_features
        )

        # Final safety: enforce finite centroids
        full_centroids = np.nan_to_num(full_centroids, nan=0.0, posinf=1e6, neginf=-1e6)

        logger.info(
            f"Calculated {len(full_centroids)} initial centroids for GMM (dimension: {feature_dim})"
        )
        return full_centroids

    def _ring_spiral_order_unified(self, color_grade: int):
        """Generate ring-spiral triplet indices in unified order (ch1,ch2,ch4)."""
        S = color_grade - 1
        order = []
        for r in range(0, S + 1):
            S_prime = S - 3 * r
            if S_prime < 0:
                break
            ring_points = []
            if S_prime == 0:
                ring_points.append((0, 0, 0))
            else:
                edge_AB = [(S_prime - t, t, 0) for t in range(0, S_prime + 1)]  # ch1->ch2
                edge_BC = [(0, S_prime - t, t) for t in range(0, S_prime + 1)]  # ch2->ch4
                edge_CA = [(t, 0, S_prime - t) for t in range(0, S_prime + 1)]  # ch4->ch1
                seq = edge_AB + edge_BC + edge_CA
                seen = set()
                for p in seq:
                    if p not in seen:
                        seen.add(p)
                        ring_points.append(p)
            ring_points_orig = [(a + r, b + r, c + r) for (a, b, c) in ring_points]
            order.extend(ring_points_orig)
        order = [t for t in order if sum(t) == (color_grade - 1) and all(0 <= v <= (color_grade - 1) for v in t)]
        out, seen = [], set()
        for t in order:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _apply_centroid_constraints(
        self,
        centroids: np.ndarray,
        sample_features: np.ndarray,
        constraint_range: float = 0.5,
    ) -> np.ndarray:
        """
        Apply constraints to centroids to prevent them from moving too far from initial positions.

        Args:
            centroids: Initial centroids
            sample_features: Sample data for calculating feature ranges
            constraint_range: Maximum allowed deviation from initial position (as fraction of feature range)

        Returns:
            Constrained centroids
        """
        constrained_centroids = centroids.copy()

        # Calculate feature ranges from sample data
        feature_ranges = np.ptp(sample_features, axis=0)  # Peak-to-peak (max - min)

        # Apply constraints to each centroid
        for i in range(len(centroids)):
            for j in range(centroids.shape[1]):
                # Calculate allowed range around initial position
                initial_pos = centroids[i, j]
                max_deviation = feature_ranges[j] * constraint_range

                # Constrain the centroid to be within the allowed range
                min_allowed = initial_pos - max_deviation
                max_allowed = initial_pos + max_deviation

                # Ensure centroid stays within bounds
                constrained_centroids[i, j] = np.clip(
                    constrained_centroids[i, j], min_allowed, max_allowed
                )

        return constrained_centroids

    def _find_histogram_maxima(
        self, data: pd.Series, expected_count: int, bw_adjust: float = 1.0
    ) -> np.ndarray:
        """Find histogram maxima using KDE (from legacy code)."""
        from scipy.signal import argrelextrema
        from scipy import stats
        import matplotlib.pyplot as plt

        logger.info(
            f"Finding histogram maxima for {len(data)} data points, expected {expected_count} maxima"
        )

        # Clean data: remove inf and NaN values
        clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) == 0:
            logger.warning("All data points are inf or NaN, using fallback")
            return np.array([])

        # For large datasets, use a more efficient approach
        if len(clean_data) > 50000:
            # Use scipy's gaussian_kde directly for better performance
            try:
                kde = stats.gaussian_kde(clean_data, bw_method=bw_adjust)
                x_range = np.linspace(clean_data.min(), clean_data.max(), 1000)
                kde_values = kde(x_range)

                # Find local maxima
                extrema_indices = argrelextrema(kde_values, np.greater)[0]
                extrema_values = [x_range[i] for i in extrema_indices]

                logger.info(f"Found {len(extrema_values)} maxima using scipy KDE")
            except Exception as e:
                logger.warning(f"KDE failed for large dataset: {e}, using fallback")
                extrema_values = []
        else:
            # For smaller datasets, use the original seaborn approach
            try:
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(1, 1))
                sns.histplot(
                    clean_data,
                    bins=100,
                    stat="count",
                    edgecolor="white",
                    alpha=1,
                    ax=ax,
                    kde=True,
                    kde_kws={"bw_adjust": bw_adjust},
                )

                # Get KDE values
                if ax.get_lines():
                    y = ax.get_lines()[0].get_ydata()
                    extrema_indices = argrelextrema(np.array(y), np.greater)[0]
                    extrema_values = [
                        float(
                            _ / len(y) * (clean_data.max() - clean_data.min())
                            + clean_data.min()
                        )
                        for _ in extrema_indices
                    ]
                else:
                    extrema_values = []

                plt.close(fig)
                logger.info(f"Found {len(extrema_values)} maxima using seaborn KDE")
            except Exception as e:
                logger.warning(f"Seaborn KDE failed: {e}, using fallback")
                extrema_values = []

        # If we don't find enough maxima, create evenly spaced ones
        # Clean and bound maxima
        vals = np.array(extrema_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(clean_data.min()), float(clean_data.max())
            vals = np.clip(vals, lo, hi)
        if len(vals) != expected_count:
            logger.info(
                f"Not enough maxima found ({len(vals)}), creating evenly spaced ones"
            )
            if expected_count <= 1:
                vals = np.array([float(clean_data.median())])
            else:
                vals = np.linspace(float(clean_data.min()), float(clean_data.max()), expected_count)

        return vals

    # Note: _reorder_centroids deprecated. Initialization now generates the desired spiral order directly.

    def _predict_layered_gmm(
        self, features: np.ndarray, processed_data: pd.DataFrame
    ) -> tuple:
        """Make predictions using layered GMM with cross-layer probability calculation."""
        labels = np.zeros(len(features), dtype=int)
        total_clusters = self.num_per_layer * self.g_layer_num
        probabilities = np.zeros((len(features), total_clusters))
        centroids = {}

        label_offset = 0

        for layer in range(self.g_layer_num):
            layer_mask = (
                processed_data.get("G_layer", np.zeros(len(processed_data))) == layer
            )

            logger.info(
                f"Layer {layer}: {np.sum(layer_mask)} points, model exists: {layer in self.model}"
            )

            if layer_mask.any() and layer in self.model:
                layer_features = features[layer_mask]
                layer_labels = self.model[layer].predict(layer_features)
                layer_probs = self.model[layer].predict_proba(layer_features)

                logger.info(
                    f"Layer {layer} prediction: {len(np.unique(layer_labels))} unique labels: {np.unique(layer_labels)}"
                )

                # Adjust labels to be globally unique (starting from 1, like legacy code)
                labels[layer_mask] = layer_labels + label_offset + 1

                # Store probabilities for this layer's clusters
                if layer_probs.shape[1] == self.num_per_layer:
                    probabilities[
                        layer_mask, label_offset : label_offset + self.num_per_layer
                    ] = layer_probs

                # Store centroids
                centroids[layer] = self.model[layer].means_

                label_offset += self.num_per_layer

        # Calculate cross-layer probabilities for all points (vectorized by layer)
        # Each point gets probability scores for all 30 clusters
        logger.info("Calculating cross-layer probabilities (vectorized)...")
        cross_layer_start = time.time()
        for layer in range(self.g_layer_num):
            if layer in self.model:
                layer_probs_full = self.model[layer].predict_proba(features)  # shape: (N, num_per_layer)
                start_idx = layer * self.num_per_layer
                end_idx = start_idx + self.num_per_layer
                probabilities[:, start_idx:end_idx] = layer_probs_full
        cross_layer_time = time.time() - cross_layer_start
        logger.info(
            f"Cross-layer probability calculation took: {cross_layer_time:.2f} seconds"
        )

        logger.info(
            f"Final labels: {len(np.unique(labels))} unique labels: {np.unique(labels)}"
        )
        return labels, probabilities, centroids

    def _predict_single_gmm(self, features: np.ndarray) -> tuple:
        """Make predictions using single GMM."""
        labels = self.model.predict(features) + 1  # Start from 1, like legacy code
        probabilities = self.model.predict_proba(features)
        centroids = self.model.means_

        return labels, probabilities, centroids

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        names = []

        if "ratios" in self.feature_types:
            ratio_names = ["ch1/A", "ch2/A", "ch4/A"]  # Color channels
            if self.include_g_channel:
                ratio_names.append("ch3/A")  # Layering channel
            names.extend(ratio_names)

        if "projections" in self.feature_types:
            names.extend(["color_dim_1", "color_dim_2"])

        if "intensity_features" in self.feature_types:
            names.extend(["sum", "log_sum"])
            for channel in ["ch1", "ch2", "ch3", "ch4"]:
                names.append(f"{channel}")

        if "statistical_features" in self.feature_types:
            names.extend(["ch1/A", "ch2/A", "ch4/A"])  # Color channels
            names.extend(["ch1/ch2_ratio", "ch4/ch1_ratio", "ch2/ch4_ratio"])

        return names

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            "method": "GMM",
            "is_fitted": self.is_fitted,
            "use_layers": self.use_layers,
            "num_per_layer": self.num_per_layer,
            "g_layer_num": self.g_layer_num,
            "covariance_type": self.covariance_type,
            "feature_names": self.get_feature_names(),
        }

        if self.is_fitted:
            if self.use_layers:
                info["layers"] = list(self.model.keys())
                info["total_components"] = sum(
                    len(gmm.means_) for gmm in self.model.values()
                )
            else:
                info["total_components"] = len(self.model.means_)

        return info
