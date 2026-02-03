"""
Configuration file loader for spot detection framework
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration file with support for modular configuration loading

    Args:
        config_path: Configuration file path, if None uses default configuration

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default configuration
        default_config_path = (
            Path(__file__).parent.parent.parent / "configs" / "spot_detection.yaml"
        )
        config_path = default_config_path

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check if this is a modular configuration
    if "config_modules" in config:
        config = load_modular_config(config_path.parent, config)

    return config


def load_modular_config(
    config_dir: Path, main_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load modular configuration by combining multiple config files

    Args:
        config_dir: Directory containing configuration files
        main_config: Main configuration dictionary with config_modules

    Returns:
        Merged configuration dictionary
    """
    merged_config = {}

    # Load each module
    for module_name in main_config.get("config_modules", []):
        module_path = config_dir / module_name
        if not module_path.exists():
            logger.warning(f"Configuration module not found: {module_path}")
            continue

        logger.info(f"Loading configuration module: {module_name}")
        with open(module_path, "r", encoding="utf-8") as f:
            module_config = yaml.safe_load(f)

        # Merge module configuration
        merged_config = merge_configs(merged_config, module_config)

    # Apply overrides
    if "overrides" in main_config:
        logger.info("Applying configuration overrides")
        merged_config = merge_configs(merged_config, main_config["overrides"])

    return merged_config


def load_batch_config(batch_config_path: str = None) -> Dict[str, Any]:
    """
    Load batch processing configuration file

    Args:
        batch_config_path: Batch configuration file path

    Returns:
        Batch configuration dictionary
    """
    if batch_config_path is None:
        default_batch_path = (
            Path(__file__).parent.parent.parent / "local" / "batch_run_ids.yaml"
        )
        batch_config_path = default_batch_path

    batch_config_path = Path(batch_config_path)

    if not batch_config_path.exists():
        raise FileNotFoundError(
            f"Batch configuration file not found: {batch_config_path}"
        )

    with open(batch_config_path, "r", encoding="utf-8") as f:
        batch_config = yaml.safe_load(f)

    return batch_config


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configurations, override_config will override same keys in base_config

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config.copy()

    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_spot_detection_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load spot detection specific configuration

    Args:
        config_path: Configuration file path

    Returns:
        Spot detection configuration dictionary
    """
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent / "configs" / "spot_detection.yaml"
        )

    return load_config(config_path)


def create_config_from_modules(
    modules: List[str], overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create configuration by loading specific modules

    Args:
        modules: List of module names to load
        overrides: Optional overrides to apply

    Returns:
        Configuration dictionary
    """
    config_dir = Path(__file__).parent.parent.parent / "configs"
    merged_config = {}

    for module_name in modules:
        module_path = config_dir / module_name
        if not module_path.exists():
            logger.warning(f"Configuration module not found: {module_path}")
            continue

        with open(module_path, "r", encoding="utf-8") as f:
            module_config = yaml.safe_load(f)

        merged_config = merge_configs(merged_config, module_config)

    if overrides:
        merged_config = merge_configs(merged_config, overrides)

    return merged_config
