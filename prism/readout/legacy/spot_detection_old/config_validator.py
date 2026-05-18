"""
Configuration validation utilities for spot detection framework
"""

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validator for spot detection parameters"""

    def __init__(self):
        self.required_sections = [
            "base",
            "detection_method",
            "intensity_extraction",
            "image_processing",
            "thresholds",
            "batch",
            "signal_processing",
            "channels",
        ]

        self.required_base_params = [
            "channels",
            "base_dir",
            "prism_panel",
            "max_memory",
            "use_tiling",
        ]
        self.valid_prism_panels = [
            "PRISM30",
            "PRISM31",
            "PRISM45",
            "PRISM46",
            "PRISM63",
            "PRISM64",
        ]
        self.valid_detection_types = ["traditional", "deep_learning"]
        self.valid_intensity_methods = [
            "direct",
            "gaussian",
            "mask",
            "integrated",
            "adaptive",
            "multiscale",
        ]
        self.valid_background_methods = ["mean", "median", "min"]

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration dictionary

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required sections
        for section in self.required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        if errors:
            return errors

        # Validate base section
        errors.extend(self._validate_base(config["base"]))

        # Validate detection method
        errors.extend(self._validate_detection_method(config["detection_method"]))

        # Validate intensity extraction
        errors.extend(
            self._validate_intensity_extraction(config["intensity_extraction"])
        )

        # Validate image processing
        errors.extend(self._validate_image_processing(config["image_processing"]))

        # Validate channels
        errors.extend(self._validate_channels(config["channels"]))

        return errors

    def _validate_base(self, base_config: Dict[str, Any]) -> List[str]:
        """Validate base configuration section"""
        errors = []

        for param in self.required_base_params:
            if param not in base_config:
                errors.append(f"Missing required base parameter: {param}")

        if "prism_panel" in base_config:
            if base_config["prism_panel"] not in self.valid_prism_panels:
                errors.append(
                    f"Invalid prism_panel: {base_config['prism_panel']}. "
                    f"Must be one of {self.valid_prism_panels}"
                )

        if "max_memory" in base_config:
            if (
                not isinstance(base_config["max_memory"], (int, float))
                or base_config["max_memory"] <= 0
            ):
                errors.append("max_memory must be a positive number")

        if "channels" in base_config:
            if (
                not isinstance(base_config["channels"], list)
                or len(base_config["channels"]) == 0
            ):
                errors.append("channels must be a non-empty list")

        return errors

    def _validate_detection_method(self, detection_config: Dict[str, Any]) -> List[str]:
        """Validate detection method configuration"""
        errors = []

        if "type" not in detection_config:
            errors.append("Missing detection_method.type")
        elif detection_config["type"] not in self.valid_detection_types:
            errors.append(
                f"Invalid detection_method.type: {detection_config['type']}. "
                f"Must be one of {self.valid_detection_types}"
            )

        # Validate method-specific parameters
        method_type = detection_config.get("type")
        if method_type == "traditional" and "traditional" in detection_config:
            errors.extend(
                self._validate_traditional_params(detection_config["traditional"])
            )
        elif method_type == "deep_learning" and "deep_learning" in detection_config:
            errors.extend(
                self._validate_deep_learning_params(detection_config["deep_learning"])
            )

        return errors

    def _validate_traditional_params(
        self, traditional_config: Dict[str, Any]
    ) -> List[str]:
        """Validate traditional method parameters"""
        errors = []

        # Check numeric parameters
        numeric_params = [
            "tophat_kernel_size",
            "tophat_break",
            "min_distance",
            "local_max_thre",
            "snr_threshold",
            "distance_threshold",
        ]

        for param in numeric_params:
            if param in traditional_config:
                if not isinstance(traditional_config[param], (int, float)):
                    errors.append(f"traditional.{param} must be a number")
                elif (
                    param in ["tophat_kernel_size", "min_distance"]
                    and traditional_config[param] <= 0
                ):
                    errors.append(f"traditional.{param} must be positive")

        # Check boolean parameters
        boolean_params = ["check_snr", "remove_duplicates"]
        for param in boolean_params:
            if param in traditional_config:
                if not isinstance(traditional_config[param], bool):
                    errors.append(f"traditional.{param} must be a boolean")

        return errors

    def _validate_deep_learning_params(self, dl_config: Dict[str, Any]) -> List[str]:
        """Validate deep learning method parameters"""
        errors = []

        # Check required parameters
        required_params = ["model_path", "model_name"]
        for param in required_params:
            if param not in dl_config or not dl_config[param]:
                errors.append(f"deep_learning.{param} is required")

        # Check numeric parameters
        numeric_params = ["prob_thresh", "nms_thresh", "roi_size"]
        for param in numeric_params:
            if param in dl_config:
                if not isinstance(dl_config[param], (int, float)):
                    errors.append(f"deep_learning.{param} must be a number")
                elif param in ["prob_thresh", "nms_thresh"] and not (
                    0 <= dl_config[param] <= 1
                ):
                    errors.append(f"deep_learning.{param} must be between 0 and 1")
                elif param == "roi_size" and dl_config[param] <= 0:
                    errors.append(f"deep_learning.{param} must be positive")

        return errors

    def _validate_intensity_extraction(
        self, intensity_config: Dict[str, Any]
    ) -> List[str]:
        """Validate intensity extraction configuration"""
        errors = []

        if "method" not in intensity_config:
            errors.append("Missing intensity_extraction.method")
        elif intensity_config["method"] not in self.valid_intensity_methods:
            errors.append(
                f"Invalid intensity_extraction.method: {intensity_config['method']}. "
                f"Must be one of {self.valid_intensity_methods}"
            )

        # Validate method-specific parameters
        method = intensity_config.get("method")
        if method in intensity_config:
            method_config = intensity_config[method]
            if method == "mask" and "background_method" in method_config:
                if (
                    method_config["background_method"]
                    not in self.valid_background_methods
                ):
                    errors.append(
                        f"Invalid background_method: {method_config['background_method']}. "
                        f"Must be one of {self.valid_background_methods}"
                    )

        return errors

    def _validate_image_processing(self, image_config: Dict[str, Any]) -> List[str]:
        """Validate image processing configuration"""
        errors = []

        # Check numeric parameters
        numeric_params = ["tophat_kernel_size", "tophat_break"]
        for param in numeric_params:
            if param in image_config:
                if (
                    not isinstance(image_config[param], (int, float))
                    or image_config[param] <= 0
                ):
                    errors.append(f"image_processing.{param} must be a positive number")

        # Check channel thresholds
        if "local_max_abs_thre_ch" in image_config:
            thre_config = image_config["local_max_abs_thre_ch"]
            if not isinstance(thre_config, dict):
                errors.append(
                    "image_processing.local_max_abs_thre_ch must be a dictionary"
                )
            else:
                for channel, threshold in thre_config.items():
                    if not isinstance(threshold, (int, float)) or threshold <= 0:
                        errors.append(
                            f"image_processing.local_max_abs_thre_ch.{channel} must be a positive number"
                        )

        return errors

    def _validate_channels(self, channels_config: Dict[str, Any]) -> List[str]:
        """Validate channel configuration"""
        errors = []

        # Check mapping
        if "mapping" not in channels_config:
            errors.append("Missing channels.mapping")
        elif not isinstance(channels_config["mapping"], dict):
            errors.append("channels.mapping must be a dictionary")

        # Check transformation matrix
        if "transformation_matrix" not in channels_config:
            errors.append("Missing channels.transformation_matrix")
        elif not isinstance(channels_config["transformation_matrix"], dict):
            errors.append("channels.transformation_matrix must be a dictionary")
        else:
            matrix = channels_config["transformation_matrix"]
            for channel, row in matrix.items():
                if not isinstance(row, list) or len(row) != 4:
                    errors.append(
                        f"channels.transformation_matrix.{channel} must be a list of 4 numbers"
                    )
                else:
                    for i, val in enumerate(row):
                        if not isinstance(val, (int, float)):
                            errors.append(
                                f"channels.transformation_matrix.{channel}[{i}] must be a number"
                            )

        return errors


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file

    Args:
        config_path: Path to configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If configuration file doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    validator = ConfigValidator()
    errors = validator.validate_config(config)

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    logger.info(f"Configuration loaded and validated successfully: {config_path}")
    return config


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary"""
    return {
        "base": {
            "channels": ["cy5", "TxRed", "cy3", "FAM"],
            "base_dir": "G:/spatial_data/processed",
            "prism_panel": "PRISM30",
            "max_memory": 32,
            "use_tiling": True,
        },
        "detection_method": {
            "type": "traditional",
            "traditional": {
                "tophat_kernel_size": 7,
                "tophat_break": 100,
                "min_distance": 2,
                "local_max_thre": 200,
                "intensity_thre": None,
                "snr_threshold": 8.0,
                "check_snr": False,
                "remove_duplicates": True,
                "distance_threshold": 2.0,
            },
        },
        "intensity_extraction": {
            "method": "gaussian",
            "gaussian": {"roi_size": 15, "fit_failed_fallback": True},
        },
        "image_processing": {
            "tophat_kernel_size": 7,
            "tophat_break": 100,
            "local_max_abs_thre_ch": {"cy5": 200, "TxRed": 200, "FAM": 200, "cy3": 200},
            "intensity_thre": None,
            "cal_snr": False,
        },
        # Note: Threshold filtering moved to gene calling step
        "batch": {"overlap": 500, "max_volume_factor": 8},
        "signal_processing": {
            "snr": 8,
            "neighborhood_size": 10,
            "kernel_size": 5,
            "min_distance": 2,
        },
        "channels": {
            "mapping": {"cy5": "R", "TxRed": "Ye", "cy3": "G", "FAM": "B"},
            "transformation_matrix": {
                "R": [1.0, 0.0, 0.0, 0.0],
                "Ye": [0.0, 1.0, 0.0, 0.0],
                "G": [0.0, 0.0, 2.5, 0.0],
                "B": [0.0, 0.0, -0.25, 0.75],
            },
        },
    }
