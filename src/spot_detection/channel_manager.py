"""
Channel Management System
统一的多通道管理系统

This module provides a unified approach to managing multi-channel data in spot detection.
Channels are represented as ch1, ch2, ch3, ch4... (wavelength from short to long).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Unified channel management system for spot detection

    Channels are represented as ch1, ch2, ch3, ch4... where:
    - ch1: shortest wavelength
    - ch2: second shortest wavelength
    - ch3: third shortest wavelength
    - ch4: longest wavelength
    - etc.
    """

    def __init__(self, config: Dict):
        """
        Initialize channel manager from configuration

        Args:
            config: Configuration dictionary containing channel information
        """
        self.config = config
        self.channel_config = config.get("channels", {})

        # Extract channel information
        self.unified_channels = self.channel_config.get("unified_channels", [])
        self.file_mapping = self.channel_config.get("file_mapping", {})
        self.wavelength_info = self.channel_config.get("wavelength_info", {})

        # If unified_channels is not provided, try to infer from file_mapping
        if not self.unified_channels and self.file_mapping:
            self.unified_channels = list(self.file_mapping.keys())

        # Create reverse mapping (unified -> original name from wavelength_info)
        self.reverse_mapping = {}
        for unified_ch in self.unified_channels:
            if unified_ch in self.wavelength_info:
                original_name = self.wavelength_info[unified_ch].get(
                    "original_name", unified_ch
                )
                self.reverse_mapping[unified_ch] = original_name
            else:
                self.reverse_mapping[unified_ch] = unified_ch

        # Create file mapping (unified -> filename) - direct mapping
        self.unified_file_mapping = self.file_mapping.copy()

        logger.info(
            f"ChannelManager initialized with {len(self.unified_channels)} channels"
        )
        logger.info(f"Unified channels: {self.unified_channels}")
        logger.info(f"Original channels: {list(self.reverse_mapping.values())}")

    def get_unified_channels(self) -> List[str]:
        """Get list of unified channel names (ch1, ch2, ch3, ch4...)"""
        return self.unified_channels.copy()

    def get_original_channels(self) -> List[str]:
        """Get list of original channel names"""
        return list(self.reverse_mapping.values())

    def get_channel_mapping(self) -> Dict[str, str]:
        """Get mapping from original channel names to unified names"""
        return {original: unified for unified, original in self.reverse_mapping.items()}

    def get_reverse_mapping(self) -> Dict[str, str]:
        """Get mapping from unified channel names to original names"""
        return self.reverse_mapping.copy()

    def get_file_mapping(self) -> Dict[str, str]:
        """Get mapping from unified channel names to filenames"""
        return self.unified_file_mapping.copy()

    def get_wavelength_info(self) -> Dict[str, Dict]:
        """
        Get wavelength information for each channel

        Returns:
            Dictionary with unified channel names as keys and wavelength info as values
        """
        unified_wavelength_info = {}
        for unified_ch in self.unified_channels:
            if unified_ch in self.wavelength_info:
                unified_wavelength_info[unified_ch] = self.wavelength_info[unified_ch]
            else:
                # Default wavelength info if not specified
                original_ch = self.reverse_mapping.get(unified_ch, unified_ch)
                unified_wavelength_info[unified_ch] = {
                    "wavelength_nm": None,
                    "description": f"Channel {unified_ch} ({original_ch})",
                }
        return unified_wavelength_info

    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get transformation matrix for channel correction

        Returns:
            NxN transformation matrix where N is the number of channels
        """
        transformation_config = self.channel_config.get("transformation_matrix", {})

        if not transformation_config:
            logger.warning("No transformation matrix found in config")
            return None

        # Build transformation matrix
        n_channels = len(self.unified_channels)
        matrix = np.zeros((n_channels, n_channels))

        # Fill matrix based on configuration
        for i, unified_ch in enumerate(self.unified_channels):
            if unified_ch in transformation_config:
                row = transformation_config[unified_ch]
                if len(row) == n_channels:
                    matrix[i, :] = row
                else:
                    logger.warning(
                        f"Transformation matrix row for {unified_ch} has wrong length"
                    )
                    matrix[i, i] = 1.0  # Identity for this channel
            else:
                matrix[i, i] = 1.0  # Identity for this channel

        logger.info(f"Transformation matrix shape: {matrix.shape}")
        return matrix

    def apply_transformation(self, intensity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply channel transformation matrix to intensity data

        Args:
            intensity_df: DataFrame with intensity data (columns: ch1, ch2, ch3, ch4...)

        Returns:
            DataFrame with transformed intensity data
        """
        matrix = self.get_transformation_matrix()

        if matrix is None:
            logger.warning(
                "No transformation matrix available, returning original data"
            )
            return intensity_df.copy()

        # Ensure we have the right columns
        available_channels = [
            ch for ch in self.unified_channels if ch in intensity_df.columns
        ]
        if len(available_channels) != len(self.unified_channels):
            logger.warning(
                f"Missing channels in intensity data. Expected: {self.unified_channels}, Got: {available_channels}"
            )

        # Extract intensity values
        intensity_values = intensity_df[available_channels].values

        # Apply transformation
        transformed_values = np.dot(matrix, intensity_values.T).T

        # Create result DataFrame
        result_df = intensity_df.copy()
        for i, ch in enumerate(self.unified_channels):
            if i < transformed_values.shape[1]:
                result_df[f"{ch}_corrected"] = transformed_values[:, i]

        return result_df

    def validate_transformation_matrix(self) -> bool:
        """
        Validate transformation matrix format and reasonableness

        Returns:
            True if valid, False otherwise
        """
        try:
            matrix = self.get_transformation_matrix()
            if matrix is None:
                logger.warning("No transformation matrix to validate")
                return True  # No matrix is valid (identity transformation)

            # Check matrix dimensions
            n_channels = len(self.unified_channels)
            if matrix.shape != (n_channels, n_channels):
                logger.error(
                    f"Transformation matrix has wrong dimensions: {matrix.shape}, expected ({n_channels}, {n_channels})"
                )
                return False

            # Check for finite values
            if not np.isfinite(matrix).all():
                logger.error("Transformation matrix contains non-finite values")
                return False

            # Check for reasonable values (not too large)
            if np.abs(matrix).max() > 100:
                logger.warning(
                    "Transformation matrix contains very large values (>100)"
                )

            logger.info("✓ Transformation matrix validation passed")
            return True

        except Exception as e:
            logger.error(f"Transformation matrix validation failed: {e}")
            return False

    def print_transformation_matrix(self):
        """Print detailed information about the transformation matrix"""
        matrix = self.get_transformation_matrix()

        if matrix is None:
            print("No transformation matrix configured")
            return

        print("Channel Transformation Matrix:")
        print("=" * 60)
        print("Format: [output_channels] = matrix × [input_channels]")
        print()

        # Print table header
        header = "Output\\Input".ljust(12)
        for input_ch in self.unified_channels:
            header += input_ch.rjust(8)
        print(header)
        print("-" * len(header))

        # Print matrix rows
        for i, output_ch in enumerate(self.unified_channels):
            row_str = output_ch.ljust(12)
            for j, input_ch in enumerate(self.unified_channels):
                value = matrix[i, j]
                row_str += f"{value:8.3f}"
            print(row_str)

        print()
        print("Mathematical Expressions:")
        for i, output_ch in enumerate(self.unified_channels):
            terms = []
            for j, input_ch in enumerate(self.unified_channels):
                coeff = matrix[i, j]
                if abs(coeff) > 1e-6:  # Only show non-zero terms
                    if coeff == 1:
                        terms.append(f"{input_ch}")
                    elif coeff == -1:
                        terms.append(f"-{input_ch}")
                    else:
                        terms.append(f"{coeff:.3f}×{input_ch}")

            if not terms:
                expression = "0"
            else:
                expression = " + ".join(terms).replace(" + -", " - ")

            print(f"{output_ch}_corrected = {expression}")
        print("=" * 60)

    def create_identity_matrix(self) -> np.ndarray:
        """Create identity transformation matrix (no transformation)"""
        n_channels = len(self.unified_channels)
        return np.eye(n_channels)

    def create_scaling_matrix(self, scaling_factors: Dict[str, float]) -> np.ndarray:
        """
        Create scaling-only transformation matrix

        Args:
            scaling_factors: Dictionary with channel names as keys and scaling factors as values

        Returns:
            Scaling transformation matrix
        """
        n_channels = len(self.unified_channels)
        matrix = np.eye(n_channels)

        for channel, factor in scaling_factors.items():
            if channel in self.unified_channels:
                channel_idx = self.unified_channels.index(channel)
                matrix[channel_idx, channel_idx] = factor
            else:
                logger.warning(f"Unknown channel for scaling: {channel}")

        return matrix

    def create_crosstalk_matrix(
        self, crosstalk_factors: Dict[str, float]
    ) -> np.ndarray:
        """
        Create crosstalk correction matrix

        Args:
            crosstalk_factors: Dictionary with format like {'ch4_from_ch3': 0.25}
                              meaning ch4 receives crosstalk from ch3 with factor 0.25

        Returns:
            Crosstalk correction matrix
        """
        n_channels = len(self.unified_channels)
        matrix = np.eye(n_channels)

        for key, factor in crosstalk_factors.items():
            if "_from_" in key:
                target_channel, source_channel = key.split("_from_")
                if (
                    target_channel in self.unified_channels
                    and source_channel in self.unified_channels
                ):
                    target_idx = self.unified_channels.index(target_channel)
                    source_idx = self.unified_channels.index(source_channel)
                    matrix[
                        target_idx, source_idx
                    ] = -factor  # Negative for crosstalk correction
                else:
                    logger.warning(f"Unknown channels in crosstalk factor: {key}")

        return matrix

    def apply_crosstalk_correction(
        self, intensity_df: pd.DataFrame, crosstalk_factors: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Apply crosstalk correction to intensity data

        Args:
            intensity_df: DataFrame with intensity data
            crosstalk_factors: Crosstalk correction factors

        Returns:
            DataFrame with crosstalk-corrected data
        """
        crosstalk_matrix = self.create_crosstalk_matrix(crosstalk_factors)

        # Apply crosstalk correction
        available_channels = [
            ch for ch in self.unified_channels if ch in intensity_df.columns
        ]
        intensity_values = intensity_df[available_channels].values
        corrected_values = np.dot(crosstalk_matrix, intensity_values.T).T

        # Ensure non-negative values (especially important for crosstalk correction)
        corrected_values = np.maximum(corrected_values, 0)

        # Create result DataFrame
        result_df = intensity_df.copy()
        for i, ch in enumerate(self.unified_channels):
            if i < corrected_values.shape[1]:
                result_df[f"{ch}_crosstalk_corrected"] = corrected_values[:, i]

        return result_df

    def create_intensity_dataframe(
        self, intensity_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Create intensity DataFrame from dictionary using unified channel names

        Args:
            intensity_dict: Dictionary with channel names as keys and intensity arrays as values

        Returns:
            DataFrame with unified channel names as columns
        """
        # Convert channel names to unified names
        unified_intensity_dict = {}
        for channel_key, intensities in intensity_dict.items():
            # Handle different naming patterns
            unified_ch = None

            # Pattern 1: Direct unified channel name (ch1, ch2, ch3, ch4)
            if channel_key in self.unified_channels:
                unified_ch = channel_key

            # Pattern 2: Original channel name (cy5, TxRed, cy3, FAM)
            elif channel_key in self.reverse_mapping.values():
                for unified, original in self.reverse_mapping.items():
                    if original == channel_key:
                        unified_ch = unified
                        break

            # Pattern 3: Channel with suffix (ch1_intensity, cy5_intensity, etc.)
            else:
                # Try to extract base channel name
                base_channel = None
                for suffix in ["_intensity", "_raw", "_corrected"]:
                    if channel_key.endswith(suffix):
                        base_channel = channel_key[: -len(suffix)]
                        break

                if base_channel:
                    # Check if it's a unified channel
                    if base_channel in self.unified_channels:
                        unified_ch = base_channel
                    # Check if it's an original channel
                    elif base_channel in self.reverse_mapping.values():
                        for unified, original in self.reverse_mapping.items():
                            if original == base_channel:
                                unified_ch = unified
                                break

            if unified_ch:
                unified_intensity_dict[unified_ch] = intensities
            else:
                logger.warning(f"No unified channel found for channel: {channel_key}")

        return pd.DataFrame(unified_intensity_dict)

    def get_channel_display_info(self) -> Dict[str, str]:
        """
        Get display information for each channel

        Returns:
            Dictionary with unified channel names as keys and display strings as values
        """
        display_info = {}
        wavelength_info = self.get_wavelength_info()

        for unified_ch in self.unified_channels:
            original_ch = self.reverse_mapping[unified_ch]
            wavelength_data = wavelength_info.get(unified_ch, {})
            wavelength = wavelength_data.get("wavelength_nm", "Unknown")
            description = wavelength_data.get("description", original_ch)

            display_info[unified_ch] = (
                f"{unified_ch} ({original_ch}, {wavelength}nm): {description}"
            )

        return display_info

    def validate_configuration(self) -> List[str]:
        """
        Validate channel configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check if we have channels defined
        if not self.unified_channels:
            errors.append("No unified channels defined in configuration")

        # Check file mapping
        for unified_ch in self.unified_channels:
            if unified_ch not in self.file_mapping:
                errors.append(f"No file mapping found for channel: {unified_ch}")

        # Check transformation matrix
        transformation_config = self.channel_config.get("transformation_matrix", {})
        if transformation_config:
            n_channels = len(self.unified_channels)
            for unified_ch in self.unified_channels:
                if unified_ch in transformation_config:
                    row = transformation_config[unified_ch]
                    if len(row) != n_channels:
                        errors.append(
                            f"Transformation matrix row for {unified_ch} has wrong length: {len(row)} (expected {n_channels})"
                        )

        return errors

    def print_summary(self):
        """Print channel configuration summary"""
        print("=" * 60)
        print("Channel Configuration Summary")
        print("=" * 60)
        print(f"Number of channels: {len(self.unified_channels)}")
        print()

        print("Channel Mapping:")
        for unified, original in self.reverse_mapping.items():
            print(f"  {unified} -> {original}")
        print()

        print("File Mapping:")
        for unified, filename in self.unified_file_mapping.items():
            print(f"  {unified} -> {filename}")
        print()

        print("Wavelength Information:")
        display_info = self.get_channel_display_info()
        for unified, info in display_info.items():
            print(f"  {info}")
        print()

        # Print transformation matrix if available
        matrix = self.get_transformation_matrix()
        if matrix is not None:
            print("Transformation Matrix:")
            print("  Input channels:", self.unified_channels)
            print(
                "  Output channels:",
                [f"{ch}_corrected" for ch in self.unified_channels],
            )
            print("  Matrix:")
            for i, row in enumerate(matrix):
                print(f"    {self.unified_channels[i]}: {row}")
        print("=" * 60)


def create_channel_manager_from_config(config_path: str) -> ChannelManager:
    """
    Create ChannelManager from configuration file

    Args:
        config_path: Path to configuration file

    Returns:
        ChannelManager instance
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return ChannelManager(config)


# Example configuration structure for reference
EXAMPLE_CONFIG = {
    "channels": {
        "unified_channels": ["ch1", "ch2", "ch3", "ch4"],
        "file_mapping": {
            "ch1": "cyc_1_cy5.tif",  # Shortest wavelength (670nm)
            "ch2": "cyc_1_TxRed.tif",  # Second shortest (615nm)
            "ch3": "cyc_1_cy3.tif",  # Third shortest (550nm)
            "ch4": "cyc_1_FAM.tif",  # Longest wavelength (520nm)
        },
        "wavelength_info": {
            "ch1": {
                "wavelength_nm": 670,
                "description": "Cy5 - Far Red",
                "original_name": "cy5",
            },
            "ch2": {
                "wavelength_nm": 615,
                "description": "Texas Red - Red",
                "original_name": "TxRed",
            },
            "ch3": {
                "wavelength_nm": 550,
                "description": "Cy3 - Green",
                "original_name": "cy3",
            },
            "ch4": {
                "wavelength_nm": 520,
                "description": "FAM - Green",
                "original_name": "FAM",
            },
        },
        "transformation_matrix": {
            "ch1": [1.0, 0.0, 0.0, 0.0],  # ch1 -> ch1_corrected
            "ch2": [0.0, 1.0, 0.0, 0.0],  # ch2 -> ch2_corrected
            "ch3": [0.0, 0.0, 2.5, 0.0],  # ch3 -> ch3_corrected (scaled)
            "ch4": [
                0.0,
                0.0,
                -0.25,
                0.75,
            ],  # ch4 -> ch4_corrected (crosstalk corrected)
        },
    }
}
