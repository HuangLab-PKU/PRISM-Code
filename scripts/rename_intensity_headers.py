"""
Batch-rename columns in existing intensity CSV files to the new wavelength-based naming convention.

The new naming convention uses excitation wavelength:
    r{round:02d}_ex{excitation_wavelength_nm}

Example: cyc_1_cy5 -> r01_ex628

Usage:
    python rename_intensity_headers.py --input-dir ./readout/ --pattern "intensity*.csv" --dry-run
    python rename_intensity_headers.py --input-dir ./readout/ --pattern "intensity*.csv"
    python rename_intensity_headers.py --input-dir ./readout/ --mapping-file my_mapping.yaml
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default mapping: all known legacy column names -> new wavelength names
# Grouped by target so that any legacy variant maps to the same new name.
DEFAULT_MAPPING = {
    # Cy5 channel (excitation 628 nm)
    "cyc_1_cy5": "r01_ex628",
    "cy5": "r01_ex628",
    "R": "r01_ex628",
    "ch1": "r01_ex628",
    "Scaled_R": "r01_ex628",
    # TxRed channel (excitation 586 nm)
    "cyc_1_TxRed": "r01_ex586",
    "TxRed": "r01_ex586",
    "Ye": "r01_ex586",
    "ch2": "r01_ex586",
    "Scaled_Ye": "r01_ex586",
    # Cy3 channel (excitation 531 nm)
    "cyc_1_cy3": "r01_ex531",
    "cy3": "r01_ex531",
    "G": "r01_ex531",
    "ch3": "r01_ex531",
    "Scaled_G": "r01_ex531",
    # FAM channel (excitation 485 nm)
    "cyc_1_FAM": "r01_ex485",
    "FAM": "r01_ex485",
    "B": "r01_ex485",
    "ch4": "r01_ex485",
    "Scaled_B": "r01_ex485",
}


def load_mapping(mapping_file):
    """Load a custom column mapping from a YAML file.

    Expected format::

        cyc_1_cy5: r01_ex628
        cy5: r01_ex628
        ...
    """
    with open(mapping_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rename_csv_headers(csv_path, mapping, dry_run=False):
    """Rename columns in a single CSV file according to *mapping*.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    mapping : dict
        {old_column_name: new_column_name, ...}
    dry_run : bool
        If True, only log what would change without writing.

    Returns
    -------
    int
        Number of columns renamed.
    """
    df = pd.read_csv(csv_path)
    renames = {col: mapping[col] for col in df.columns if col in mapping}

    if not renames:
        logger.info(f"  {csv_path.name}: no columns to rename")
        return 0

    for old, new in renames.items():
        logger.info(f"  {csv_path.name}: {old} -> {new}")

    if not dry_run:
        df = df.rename(columns=renames)
        df.to_csv(csv_path, index=False)
        logger.info(f"  {csv_path.name}: saved ({len(renames)} columns renamed)")

    return len(renames)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-rename intensity CSV column headers to wavelength naming convention"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory to search for CSV files",
    )
    parser.add_argument(
        "--pattern", type=str, default="intensity*.csv",
        help='Glob pattern for CSV files (default: "intensity*.csv")',
    )
    parser.add_argument(
        "--mapping-file", type=Path, default=None,
        help="Optional YAML file with custom column mapping (overrides built-in default)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing",
    )
    args = parser.parse_args()

    mapping = DEFAULT_MAPPING
    if args.mapping_file:
        mapping = load_mapping(args.mapping_file)
        logger.info(f"Loaded custom mapping from {args.mapping_file} ({len(mapping)} entries)")

    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        logger.warning(f"No files matching '{args.pattern}' in {args.input_dir}")
        return

    mode = "DRY RUN" if args.dry_run else "RENAME"
    logger.info(f"[{mode}] Found {len(csv_files)} file(s) in {args.input_dir}")

    total_renamed = 0
    for csv_path in csv_files:
        total_renamed += rename_csv_headers(csv_path, mapping, dry_run=args.dry_run)

    logger.info(f"Done. Total columns renamed: {total_renamed}")


if __name__ == "__main__":
    main()
