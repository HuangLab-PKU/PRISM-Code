"""
Configuration loading utilities for gene calling framework.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


def load_gene_calling_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load gene calling configuration from YAML file.

    Supports modular configuration loading similar to spot detection.

    Args:
        config_path: Path to main configuration file

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load main configuration
    with open(config_path, "r", encoding="utf-8") as f:
        main_config = yaml.safe_load(f)

    # Get config directory
    config_dir = config_path.parent

    # Load configuration modules
    config_modules = main_config.get("config_modules", [])
    merged_config = {}

    for module_name in config_modules:
        module_path = config_dir / module_name
        if module_path.exists():
            with open(module_path, "r", encoding="utf-8") as f:
                module_config = yaml.safe_load(f)
            merged_config = _merge_configs(merged_config, module_config)
            logger.info(f"Loaded configuration module: {module_name}")
        else:
            logger.warning(f"Configuration module not found: {module_path}")

    # Apply overrides
    overrides = main_config.get("overrides", {})
    if overrides:
        merged_config = _merge_configs(merged_config, overrides)
        logger.info("Applied configuration overrides")

    return merged_config


def _merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_gene_calling_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate gene calling configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required sections
    required_sections = ["data", "feature_extraction", "classification"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")

    # Validate classification method
    if "classification" in config:
        method = config["classification"].get("method")
        if not method:
            errors.append("Classification method not specified")
        elif method not in ["gmm"]:  # Add more methods as they're implemented
            errors.append(f"Unsupported classification method: {method}")

    # Validate GMM configuration if method is GMM
    if config.get("classification", {}).get("method") == "gmm":
        gmm_config = config.get("classification", {}).get("gmm", {})
        if not gmm_config:
            errors.append("GMM configuration missing")
        else:
            # Check required GMM parameters (num_per_layer is now in base config)
            required_gmm_params = ["covariance_type"]
            for param in required_gmm_params:
                if param not in gmm_config:
                    errors.append(f"Missing required GMM parameter: {param}")

        # Check PRISM panel parameters in base config
        prism_config = config.get("prism_panel", {})
        if not prism_config:
            errors.append("PRISM panel configuration missing")
        else:
            required_prism_params = ["type", "num_per_layer", "g_layer_num"]
            for param in required_prism_params:
                if param not in prism_config:
                    errors.append(f"Missing required PRISM panel parameter: {param}")

    # Validate feature extraction
    if "feature_extraction" in config:
        feature_types = config["feature_extraction"].get("feature_types", [])
        valid_types = [
            "ratios",
            "projections",
            "intensity_features",
            "statistical_features",
        ]
        for feature_type in feature_types:
            if feature_type not in valid_types:
                errors.append(f"Invalid feature type: {feature_type}")

    return errors


def get_default_gene_calling_config() -> Dict[str, Any]:
    """
    Get default gene calling configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "input_format": {
                "intensity_columns": ["ch1", "ch2", "ch3", "ch4"],
                "coordinate_columns": ["Y", "X"],
            },
            "output": {
                "save_predictions": True,
                "save_model": True,
                "save_evaluation": True,
                "save_visualizations": True,
            },
        },
        "feature_extraction": {
            "feature_types": [
                "ratios",
                "projections",
                "intensity_features",
                "statistical_features",
            ],
            "include_g_channel": True,
        },
        "classification": {
            "method": "gmm",
            "gmm": {
                "covariance_type": "diag",
                "use_layers": True,
                "num_per_layer": 15,
                "max_iter": 100,
                "tol": 1e-3,
                "random_state": 42,
            },
        },
        "evaluation": {
            "visualization": {
                "figure_size": [10, 8],
                "dpi": 300,
                "save_format": "png",
                "color_scheme": "tab20",
            },
            "metrics": [
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            ],
            "generate_report": True,
            "report_format": "markdown",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }
