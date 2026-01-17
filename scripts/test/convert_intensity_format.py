#!/usr/bin/env python3
"""
Convert intensity.csv from PRISM format to standard ch1-ch4 format.

This script converts intensity data from PRISM channel names (R, Ye, G, B)
to standard channel names (ch1, ch2, ch3, ch4) for compatibility with
the gene calling pipeline.

Usage:
    python convert_intensity_format.py --input input.csv --output output.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_intensity_format(input_path: str, output_path: str):
    """
    Convert intensity data from PRISM format to standard format.

    Args:
        input_path: Path to input intensity.csv file
        output_path: Path to output intensity.csv file
    """
    logger.info(f"Reading intensity data from: {input_path}")

    # Read the intensity data
    df = pd.read_csv(input_path)

    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Original columns: {list(df.columns)}")

    # Check if we have the expected PRISM format
    expected_prism_cols = ["Y", "X", "R", "Ye", "G", "B"]
    if not all(col in df.columns for col in expected_prism_cols):
        raise ValueError(
            f"Expected columns {expected_prism_cols} not found in input file"
        )

    # Create a copy for conversion
    converted_df = df.copy()

    # Map PRISM channel names to standard ch1-ch4 names
    # Based on the corrected mapping:
    # ch1: Cy5 (670nm) -> R
    # ch2: TxRed (615nm) -> Ye
    # ch3: FAM (520nm) -> G (layering channel)
    # ch4: Cy3 (550nm) -> B
    channel_mapping = {
        "R": "ch1",  # Cy5 -> ch1
        "Ye": "ch2",  # TxRed -> ch2
        "G": "ch3",  # FAM -> ch3 (layering channel)
        "B": "ch4",  # Cy3 -> ch4
    }

    # Rename channels
    for prism_name, standard_name in channel_mapping.items():
        converted_df[standard_name] = converted_df[prism_name]

    # Keep only the standard columns: index, Y, X, ch1, ch2, ch3, ch4
    standard_columns = [df.columns[0], "Y", "X", "ch1", "ch2", "ch3", "ch4"]
    converted_df = converted_df[standard_columns]

    # Rename the first column to match expected format
    converted_df.columns = ["", "Y", "X", "ch1", "ch2", "ch3", "ch4"]

    logger.info(f"Converted data shape: {converted_df.shape}")
    logger.info(f"Converted columns: {list(converted_df.columns)}")

    # Save the converted data
    logger.info(f"Saving converted data to: {output_path}")
    converted_df.to_csv(output_path, index=False)

    # Print some statistics
    logger.info("Conversion completed successfully!")
    logger.info(f"Total data points: {len(converted_df)}")
    logger.info("Channel statistics:")
    for ch in ["ch1", "ch2", "ch3", "ch4"]:
        non_zero = (converted_df[ch] > 0).sum()
        logger.info(
            f"  {ch}: {non_zero} non-zero points ({non_zero / len(converted_df) * 100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert intensity data from PRISM to standard format"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input intensity.csv file path"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output intensity.csv file path"
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

    # Perform conversion
    convert_intensity_format(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
