#!/usr/bin/env python3
"""
Extract coordinates from intensity file to create a separate coordinates.csv file.

Usage:
    python extract_coordinates.py --input intensity.csv --output coordinates.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_coordinates(input_path: str, output_path: str):
    """
    Extract Y, X coordinates from intensity file.

    Args:
        input_path: Path to input intensity.csv file
        output_path: Path to output coordinates.csv file
    """
    logger.info(f"Reading intensity data from: {input_path}")

    # Read the intensity data
    df = pd.read_csv(input_path)

    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Original columns: {list(df.columns)}")

    # Check if we have Y and X columns
    if "Y" not in df.columns or "X" not in df.columns:
        raise ValueError("Y and X columns not found in input file")

    # Extract coordinates
    coordinates_df = df[["Y", "X"]].copy()

    logger.info(f"Extracted coordinates shape: {coordinates_df.shape}")

    # Save coordinates
    logger.info(f"Saving coordinates to: {output_path}")
    coordinates_df.to_csv(output_path, index=False)

    logger.info("Coordinates extraction completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract coordinates from intensity file"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input intensity.csv file path"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output coordinates.csv file path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert paths to absolute paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract coordinates
    extract_coordinates(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
