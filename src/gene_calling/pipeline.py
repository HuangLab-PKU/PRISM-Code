"""
Complete pipeline for signal point classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import logging
from .base import ClassificationResult
from .gmm_method import GMMMethod
from .evaluator import ClassificationEvaluator
from .config_loader import load_gene_calling_config, validate_gene_calling_config


class SignalClassificationPipeline:
    """
    Complete pipeline for signal point classification.

    Handles the full workflow from raw intensity data to classified results
    with evaluation and visualization.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the classification pipeline.

        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (overrides config_path if provided)
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_gene_calling_config(config_path)
        else:
            self.config = self._get_default_config()

        # Validate configuration
        validation_errors = validate_gene_calling_config(self.config)
        if validation_errors:
            raise ValueError(f"Configuration validation failed: {validation_errors}")

        # Initialize components
        self.method = self._create_method()
        self.evaluator = ClassificationEvaluator(self.config.get("evaluation", {}))

        # Pipeline state
        self.is_fitted = False
        self.training_data = None
        self.training_features = None

    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "preprocessing": {
                "scaling_factors": {"ch1": 1.0, "ch2": 1.0, "ch3": 1.0, "ch4": 1.0},
                "crosstalk_factor": 0.25,
                "fret_adjustments": {"G_ye_factor": 0.6, "B_g_factor": 0.1},
                "gaussian_noise_scale": 0.01,
                "prism_panel": "PRISM30",
            },
            "feature_extraction": {
                "feature_types": ["ratios", "projections", "intensity_features"],
                "include_g_channel": True,
            },
            "classification": {
                "method": "gmm",
                "gmm": {
                    "covariance_type": "diag",
                    "max_iter": 100,
                    "tol": 1e-3,
                    "n_init": 1,
                    "use_layers": True,
                    "g_layer_column": "G_layer",
                    "num_per_layer": 15,
                    "scale_features": True,
                },
            },
            "evaluation": {"visualization": {"figure_size": (10, 8), "dpi": 300}},
        }

    def _create_method(self) -> GMMMethod:
        """Create method based on configuration."""
        method = self.config.get("classification", {}).get("method", "gmm")

        if method == "gmm":
            # Combine all relevant config sections for GMM method
            gmm_config = {
                **self.config.get("preprocessing", {}),
                **self.config.get("feature_extraction", {}),
                **self.config.get("classification", {}),
            }

            # Add PRISM panel parameters from base config
            prism_config = self.config.get("prism_panel", {})
            channel_grading = prism_config.get("channel_grading", {})
            gmm_config.update(
                {
                    "prism_panel_type": prism_config.get("type", "PRISM30"),
                    "num_per_layer": prism_config.get("num_per_layer", 15),
                    "g_layer_num": prism_config.get("g_layer_num", 2),
                    "total_components": prism_config.get("total_components", 30),
                    "color_grade": channel_grading.get("color_channels", 5),
                    "layer_grade": channel_grading.get("layer_channel", 2),
                }
            )

            return GMMMethod(gmm_config)
        else:
            raise ValueError(f"Unsupported classification method: {method}")

    def load_data(
        self,
        data_path: Union[str, Path],
        coordinates_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Load signal intensity data.

        Args:
            data_path: Path to intensity CSV file
            coordinates_path: Optional path to coordinates CSV file

        Returns:
            Combined DataFrame with intensity and coordinate data
        """
        self.logger.info(f"Loading data from {data_path}")

        # Load intensity data
        intensity_data = pd.read_csv(data_path)

        # Use standard ch1-ch4 naming system directly

        # Load coordinates if provided
        if coordinates_path is not None:
            coordinates = pd.read_csv(coordinates_path)
            intensity_data = pd.concat([intensity_data, coordinates], axis=1)

        self.logger.info(f"Loaded {len(intensity_data)} signal points")
        return intensity_data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw signal data.

        Args:
            data: Raw intensity data

        Returns:
            Preprocessed data
        """
        self.logger.info("Preprocessing signal data")

        # Apply preprocessing using method's preprocess method
        processed_data = self.method.preprocess(data)

        # Apply intensity thresholds
        thre_min = self.config.get("preprocessing", {}).get("thre_min", 200)
        thre_max = self.config.get("preprocessing", {}).get("thre_max", 10000)

        if "sum" in processed_data.columns:
            mask = (processed_data["sum"] > thre_min) & (
                processed_data["sum"] < thre_max
            )
            processed_data = processed_data[mask]
            self.logger.info(
                f"Applied intensity thresholds: {len(processed_data)} points remaining"
            )

        return processed_data

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from preprocessed data.

        Args:
            data: Preprocessed data

        Returns:
            Feature matrix
        """
        self.logger.info("Extracting features")
        features = self.feature_extractor.extract_features(data)
        self.logger.info(
            f"Extracted {features.shape[1]} features for {features.shape[0]} samples"
        )
        return features

    def fit(
        self, data: pd.DataFrame, ground_truth: Optional[np.ndarray] = None
    ) -> "SignalClassificationPipeline":
        """
        Fit the classification pipeline to training data.

        Args:
            data: Training data (raw intensity data)
            ground_truth: Optional ground truth labels

        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting classification pipeline")

        # Store training data
        self.training_data = data.copy()

        # Fit method (method handles preprocessing and feature extraction internally)
        self.logger.info(f"About to call method.fit with data shape: {data.shape}")
        self.method.fit(data, ground_truth, None)
        self.logger.info("Method.fit completed successfully")
        self.is_fitted = True

        self.logger.info("Pipeline fitting completed")
        return self

    def predict(self, data: pd.DataFrame) -> ClassificationResult:
        """
        Predict labels for new data.

        Args:
            data: Input data (raw intensity data)

        Returns:
            ClassificationResult containing predictions
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")

        self.logger.info("Making predictions")

        # Make predictions (method handles preprocessing and feature extraction internally)
        result = self.method.predict(data)

        self.logger.info(f"Predicted labels for {len(result.labels)} samples")
        return result

    def evaluate(
        self,
        result: ClassificationResult,
        data: pd.DataFrame,
        ground_truth: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate classification results.

        Args:
            result: ClassificationResult object
            data: Original data
            ground_truth: Optional ground truth labels
            output_dir: Optional directory for saving visualizations

        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Evaluating classification results")

        # Perform evaluation (result.labels already correspond to processed data)
        evaluation = self.evaluator.evaluate_classification(result, data, ground_truth)

        # Generate visualizations if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            plot_paths = self.evaluator.visualize_classification(
                result, data, output_dir
            )
            evaluation["plot_paths"] = plot_paths

            # Generate channel space visualization using evaluator
            g_layer_num = self.config.get("preprocessing", {}).get("g_layer_num", 2)
            channel_space_path = self.evaluator.visualize_channel_space(
                result, data, output_dir, g_layer_num, self.method
            )
            evaluation["channel_space_plot"] = channel_space_path

            # Generate report
            report_path = self.evaluator.generate_report(
                evaluation, output_dir / "evaluation_report.md"
            )
            evaluation["report_path"] = report_path

        self.logger.info("Evaluation completed")
        return evaluation

    def run_full_pipeline(
        self,
        data_path: Union[str, Path],
        coordinates_path: Optional[Union[str, Path]] = None,
        ground_truth: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete classification pipeline.

        Args:
            data_path: Path to intensity data
            coordinates_path: Optional path to coordinates
            ground_truth: Optional ground truth labels
            output_dir: Optional output directory for results

        Returns:
            Dictionary containing all results
        """
        self.logger.info("Running full classification pipeline")

        # Load data
        data = self.load_data(data_path, coordinates_path)

        # Fit pipeline
        self.fit(data, ground_truth)

        # Make predictions
        result = self.predict(data)

        # Evaluate results
        evaluation = self.evaluate(result, data, ground_truth, output_dir)

        # Prepare results
        results = {
            "classification_result": result,
            "evaluation": evaluation,
            "config": self.config,
            "model_info": self.method.get_model_info(),
        }

        # Save results if output directory provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save predictions
            result_df = result.to_dataframe(data)
            result_df.to_csv(output_dir / "predictions.csv", index=False)

            # Save configuration
            with open(output_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)

        self.logger.info("Full pipeline completed")
        return results

    def save_model(self, output_path: Union[str, Path]):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "config": self.config,
            "method_info": self.method.get_model_info(),
        }

        with open(output_path, "w") as f:
            yaml.dump(model_data, f, default_flow_style=False)

    def load_model(self, model_path: Union[str, Path]):
        """Load a trained model."""
        with open(model_path, "r") as f:
            model_data = yaml.safe_load(f)

        self.config = model_data["config"]
        # Note: This is a simplified loading - in practice you'd need to
        # properly reconstruct the classifier state
        self.logger.warning(
            "Model loading is simplified - classifier state not fully restored"
        )
