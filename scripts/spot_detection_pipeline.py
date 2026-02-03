#!/usr/bin/env python3
"""
Spot Detection Pipeline
Main execution script for the unified spot detection framework
Supports multi-channel processing following multi_channel_readout.py workflow
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yaml
import shutil
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from spot_detection import (
    UnifiedSpotDetector,
    detect_spots,
    create_detector_from_config,
    create_preset_detector,
    load_spot_detection_config,
    create_config_from_modules,
    ChannelManager,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file"""
    try:
        from tifffile import imread

        image = imread(image_path)
        logger.info(f"Loaded image: {image_path}, shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def load_image_with_memmap(image_path: str, use_memmap: bool = True) -> np.ndarray:
    """
    Load image from file with optional memory mapping

    Args:
        image_path: Path to the image file
        use_memmap: Whether to use memory mapping for large images

    Returns:
        Image array
    """
    try:
        from tifffile import imread
        from spot_detection import estimate_memory_usage

        # Get image info first
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()
            shape = image.shape
            dtype = image.dtype

        # Estimate memory usage
        memory_info = estimate_memory_usage(shape, dtype, num_channels=1)

        # Use memmap for large images (> 2GB)
        if use_memmap and memory_info["original_memory_gb"] > 2.0:
            logger.info(
                f"Large image detected ({memory_info['original_memory_gb']:.2f}GB), using memory mapping"
            )
            # For now, still load normally but log the recommendation
            # In future, this could be integrated with MemmapImageLoader
            logger.warning(
                "Memory mapping not yet integrated in pipeline - loading normally"
            )

        logger.info(
            f"Loaded image: {image_path}, shape: {image.shape}, memory: {memory_info['original_memory_gb']:.2f}GB"
        )
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def load_multi_channel_images(image_dir: str, channel_manager: ChannelManager) -> dict:
    """Load multi-channel images from directory using channel manager"""
    image_dict = {}
    image_dir = Path(image_dir)

    # Get file mapping from channel manager
    file_mapping = channel_manager.get_file_mapping()

    for unified_channel, filename in file_mapping.items():
        file_path = image_dir / filename
        if file_path.exists():
            image_dict[unified_channel] = load_image(str(file_path))
            raw_channel = channel_manager.get_reverse_mapping()[unified_channel]
            logger.info(f"Loaded {unified_channel} ({raw_channel}) from: {filename}")
        else:
            logger.warning(f"File not found for channel {unified_channel}: {filename}")

    return image_dict


def copy_config_to_output(config_path: str, output_dir: str, full_config: dict = None):
    """Copy configuration file to output directory with spot_detection prefix"""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original config file with spot_detection prefix
    config_name = config_path.name
    if not config_name.startswith("spot_detection_"):
        config_name = f"spot_detection_{config_name}"

    output_config_path = output_dir / config_name
    shutil.copy2(config_path, output_config_path)
    logger.info(f"Copied config to: {output_config_path}")

    # Also save the full merged configuration with spot_detection prefix
    if full_config is not None:
        full_config_path = output_dir / "spot_detection_full_config.yaml"
        with open(full_config_path, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved full merged config to: {full_config_path}")

    return output_config_path


def process_multi_channel_spot_detection(
    image_dict: dict, detector: UnifiedSpotDetector, output_dir: str, config: dict
):
    """
    Process multi-channel spot detection using the unified detector architecture
    """
    logger.info("Processing multi-channel spot detection")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize channel manager
    channel_manager = ChannelManager(config)

    # Validate channel configuration
    validation_errors = channel_manager.validate_configuration()
    if validation_errors:
        logger.error("Channel configuration validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return False

    # Print channel configuration summary
    channel_manager.print_summary()

    # Use the unified detector's multi-channel detection capability
    # The detector will handle all the complex logic based on config
    result = detector.detect_spots(image_dict)

    logger.info(f"Detected {len(result.coordinates)} spots across all channels")

    # Apply channel transformations using ChannelManager
    logger.info("Applying channel transformations")

    # Convert intensities dict to DataFrame if needed
    if result.intensities is not None and isinstance(result.intensities, dict):
        # Convert dict to DataFrame using channel manager
        intensity_df = channel_manager.create_intensity_dataframe(result.intensities)
    elif result.intensities is not None and isinstance(
        result.intensities, pd.DataFrame
    ):
        intensity_df = result.intensities.copy()
    else:
        logger.warning("No intensity data found")
        return True

    logger.info(f"Available columns: {list(intensity_df.columns)}")

    # Create raw intensity DataFrame (before transformation)
    # Use unified channel names (ch1, ch2, ch3, ch4...)
    unified_channels = channel_manager.get_unified_channels()
    raw_intensity_df = intensity_df[unified_channels].copy()

    # Apply transformation using ChannelManager
    transformed_intensity_df = channel_manager.apply_transformation(intensity_df)

    # Extract only the corrected columns for the final output
    corrected_columns = [f"{ch}_corrected" for ch in unified_channels]
    if all(col in transformed_intensity_df.columns for col in corrected_columns):
        final_transformed_df = transformed_intensity_df[corrected_columns].copy()
        # Rename corrected columns to standard names
        final_transformed_df.columns = unified_channels
    else:
        logger.warning("Some corrected columns missing, using raw data")
        final_transformed_df = raw_intensity_df.copy()

    # Save raw intensity file
    raw_output_path = output_dir / "intensity_raw.csv"
    raw_intensity_df.to_csv(raw_output_path, index=False)
    logger.info(f"Raw intensity saved to: {raw_output_path}")

    # Save transformed intensity file
    transformed_output_path = output_dir / "intensity.csv"
    final_transformed_df.to_csv(transformed_output_path, index=False)
    logger.info(f"Transformed intensity saved to: {transformed_output_path}")

    # Also save coordinates separately
    coords_df = pd.DataFrame(
        {"Y": result.coordinates[:, 0], "X": result.coordinates[:, 1]}
    )
    coords_output_path = output_dir / "coordinates.csv"
    coords_df.to_csv(coords_output_path, index=False)
    logger.info(f"Coordinates saved to: {coords_output_path}")

    # Save metadata as YAML
    metadata_output_path = output_dir / "metadata.yaml"
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        yaml.dump(result.metadata, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Metadata saved to: {metadata_output_path}")

    # Save channel information
    channel_info = {
        "unified_channels": channel_manager.get_unified_channels(),
        "original_channels": channel_manager.get_original_channels(),
        "wavelength_info": channel_manager.get_wavelength_info(),
        "file_mapping": channel_manager.get_file_mapping(),
    }
    channel_info_path = output_dir / "channel_info.yaml"
    with open(channel_info_path, "w", encoding="utf-8") as f:
        yaml.dump(channel_info, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Channel information saved to: {channel_info_path}")

    logger.info("Multi-channel spot detection completed successfully!")
    return True


def save_results(result, output_dir: str, prefix: str = "spot_detection"):
    """Save detection results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save coordinates
    coords_file = output_dir / f"{prefix}_coordinates.csv"
    coords_df = pd.DataFrame(
        {"Y": result.coordinates[:, 0], "X": result.coordinates[:, 1]}
    )
    coords_df.to_csv(coords_file, index=False)
    logger.info(f"Saved coordinates to: {coords_file}")

    # Save intensities
    if result.intensities:
        intensity_file = output_dir / f"{prefix}_intensities.csv"
        intensity_df = pd.DataFrame(result.intensities)
        intensity_df.to_csv(intensity_file, index=False)
        logger.info(f"Saved intensities to: {intensity_file}")

    # Save metadata
    metadata_file = output_dir / f"{prefix}_metadata.csv"
    metadata_df = pd.DataFrame([result.metadata])
    metadata_df.to_csv(metadata_file, index=False)
    logger.info(f"Saved metadata to: {metadata_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Spot Detection Pipeline")

    # Input options
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input image directory (e.g., stitched directory)",
    )
    parser.add_argument("--run-id", required=True, help="Run ID for processing")
    parser.add_argument(
        "--data-dir", default="G:/spatial_data", help="Base data directory"
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory (default: {data_dir}/processed/{run_id}/readout)",
    )

    # Configuration options
    parser.add_argument("--config", required=True, help="Configuration file path")

    # Processing options
    parser.add_argument(
        "--multi-channel",
        action="store_true",
        default=True,
        help="Process as multi-channel (default: True)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        config = load_spot_detection_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")

        # Create detector
        detector = UnifiedSpotDetector(config)

        # Set up paths
        data_dir = Path(args.data_dir)
        processed_dir = data_dir / "processed"
        run_dir = processed_dir / args.run_id
        input_dir = run_dir / "stitched"

        # Set output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = run_dir / "readout"

        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Copy config to output directory
        copy_config_to_output(args.config, output_dir, config)

        # Initialize channel manager
        channel_manager = ChannelManager(config)

        # Validate channel configuration
        validation_errors = channel_manager.validate_configuration()
        if validation_errors:
            logger.error("Channel configuration validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return 1

        # Get channel file mapping from channel manager
        file_mapping = channel_manager.get_file_mapping()
        logger.info(f"Channel file mapping: {file_mapping}")

        # Load multi-channel images
        image_dict = load_multi_channel_images(str(input_dir), channel_manager)
        if not image_dict:
            logger.error("No images found in input directory")
            return 1

        logger.info(f"Loaded images for channels: {list(image_dict.keys())}")

        # Process multi-channel spot detection
        success = process_multi_channel_spot_detection(
            image_dict, detector, str(output_dir), config
        )

        if success:
            logger.info("=" * 50)
            logger.info("Pipeline completed successfully!")
            logger.info(f"  Run ID: {args.run_id}")
            logger.info(f"  Channels processed: {list(image_dict.keys())}")
            logger.info(f"  Output directory: {output_dir}")
            logger.info("=" * 50)
            return 0
        else:
            logger.error("Pipeline failed")
            return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
