"""
Gene calling pipeline for PRISM.

This pipeline performs signal point classification:
- Input: intensity.csv (from readout.py, with excitation-wavelength column names)
- Output: mapping.csv (gene assignments with top-1/top-2 and probabilities)
         intensity_corrected.csv (corrected intensities with fluorophore names)

Channel correction (crosstalk, scaling, FRET) is handled by the pipeline
preprocess step via a 4x4 unmixing matrix defined in the config.

The script uses the gene_calling pipeline from src/gene_calling.
"""

import os
import shutil
from pathlib import Path

import logging
import numpy as np
import pandas as pd

from prism.gene_calling.pipeline import SignalClassificationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ========== Configuration ==========
BASE_DIR = Path("path_to_processed_dataset")
RUN_ID = "example_data"

# Paths
src_dir = BASE_DIR / RUN_ID
read_dir = src_dir / "readout"
config_dir = Path("path_to_configs")

# Input / Output files
INTENSITY_FILE = read_dir / "intensity.csv"
POSITION_FILE = read_dir / "position.csv"  # optional, for validation
MAPPING_FILE = read_dir / "mapping.csv"
CORRECTED_FILE = read_dir / "intensity_corrected.csv"
CONFIG_FILE = config_dir / "gene_calling_base.yaml"


def main(
    intensity_file=INTENSITY_FILE,
    position_file=POSITION_FILE,
    mapping_file=MAPPING_FILE,
    corrected_file=CORRECTED_FILE,
    config_file=None,
):
    """
    Main function for gene calling.

    Parameters
    ----------
    intensity_file : Path
        Path to intensity.csv (from readout.py, with excitation-wavelength columns).
    position_file : Path, optional
        Path to position.csv (for validation, not required).
    mapping_file : Path
        Path to output mapping.csv.
    corrected_file : Path
        Path to output intensity_corrected.csv.
    config_file : Path, optional
        Path to YAML configuration file.  If None, uses default config.
    """
    logger.info("=" * 80)
    logger.info("PRISM Gene Calling Pipeline")
    logger.info("=" * 80)
    logger.info(f"RUN_ID: {RUN_ID}")
    logger.info(f"Input:  {intensity_file}")
    logger.info(f"Output: {mapping_file}")
    logger.info(f"Corrected: {corrected_file}")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Load intensity data
    # ------------------------------------------------------------------
    if not Path(intensity_file).exists():
        raise FileNotFoundError(f"Intensity file not found: {intensity_file}")

    logger.info("Loading intensity data...")
    intensity_df = pd.read_csv(intensity_file)
    logger.info(f"Loaded {len(intensity_df)} signal points")

    if "index" not in intensity_df.columns:
        logger.warning("'index' column not found -- creating from row index")
        intensity_df["index"] = intensity_df.index

    # ------------------------------------------------------------------
    # Position validation (optional)
    # ------------------------------------------------------------------
    if position_file and Path(position_file).exists():
        logger.info("Loading position data for validation...")
        position_df = pd.read_csv(position_file)
        logger.info(f"Loaded {len(position_df)} positions")

        intensity_indices = set(intensity_df["index"].values)
        position_indices = set(position_df["index"].values)
        if intensity_indices != position_indices:
            common_indices = intensity_indices & position_indices
            intensity_df = intensity_df[intensity_df["index"].isin(common_indices)]
            logger.info(
                f"Index mismatch resolved: using {len(common_indices)} common indices"
            )

    # ------------------------------------------------------------------
    # Initialize pipeline (correction is done inside pipeline.fit/predict)
    # ------------------------------------------------------------------
    logger.info("Initializing classification pipeline...")

    if config_file and Path(config_file).exists():
        logger.info(f"Loading config from: {config_file}")
        pipeline = SignalClassificationPipeline(config_path=config_file)
    else:
        logger.info("Using default configuration")
        pipeline = SignalClassificationPipeline(config=_get_default_config())

    # Tell the pipeline where to save intensity_corrected.csv
    correction_cfg = pipeline.config.get("channel_correction", {})
    if correction_cfg:
        correction_cfg["output_path"] = str(corrected_file)

    # ------------------------------------------------------------------
    # Stage 1: Fit
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Stage 1: Fitting Classification Model")
    logger.info("=" * 80)
    pipeline.fit(intensity_df)
    logger.info("Model fitting completed")

    # ------------------------------------------------------------------
    # Stage 2: Predict
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Stage 2: Making Predictions")
    logger.info("=" * 80)
    result = pipeline.predict(intensity_df)
    logger.info(f"Predicted labels for {len(result.labels)} samples")

    # ------------------------------------------------------------------
    # Stage 3: Save mapping
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Stage 3: Creating Mapping File")
    logger.info("=" * 80)

    mapping_df = result.to_dataframe(intensity_df)

    # Attach the original index
    if "index" in intensity_df.columns:
        mapping_df["index"] = intensity_df["index"].values
    else:
        mapping_df["index"] = mapping_df.index

    # Put index first
    cols = ["index"] + [c for c in mapping_df.columns if c != "index"]
    mapping_df = mapping_df[cols]

    # Validate index consistency
    if "index" in intensity_df.columns:
        intensity_indices = set(intensity_df["index"].values)
        mapping_indices = set(mapping_df["index"].values)
        if intensity_indices != mapping_indices:
            common_indices = sorted(list(intensity_indices & mapping_indices))
            mapping_df = mapping_df[mapping_df["index"].isin(common_indices)].sort_values("index")
            logger.info(f"Using {len(common_indices)} common indices for mapping file")

    mapping_df.to_csv(mapping_file, index=False)
    logger.info(f"Saved mapping file: {mapping_file} ({len(mapping_df)} entries)")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)

    # Prefer gene names, fall back to numeric label
    label_col = (
        "top1_gene" if "top1_gene" in mapping_df.columns else "top1_label"
    )
    if label_col in mapping_df.columns:
        label_counts = mapping_df[label_col].value_counts()
        logger.info(f"Number of unique labels: {len(label_counts)}")
        logger.info("Top 10 labels:")
        for label, count in label_counts.head(10).items():
            logger.info(f"  {label}: {count}")

    if "top1_prob" in mapping_df.columns:
        probs = mapping_df["top1_prob"]
        logger.info(
            f"Probability stats: mean={probs.mean():.4f}, "
            f"median={probs.median():.4f}, "
            f"min={probs.min():.4f}, max={probs.max():.4f}"
        )

    logger.info("=" * 80)
    logger.info("Gene calling completed successfully!")
    logger.info(f"Output file: {mapping_file}")
    logger.info("=" * 80)

    return mapping_df


def _get_default_config():
    """Get default configuration for the gene calling pipeline.

    Channel correction uses the identity matrix by default (no correction).
    To activate correction, load a config file with the appropriate preset.
    """
    return {
        "channel_correction": {
            "channels": {
                "r01_ex628": {"fluorophore": "r01_Cy5", "excitation_nm": 628, "emission_nm": 670, "role": "color"},
                "r01_ex586": {"fluorophore": "r01_TxRed", "excitation_nm": 586, "emission_nm": 615, "role": "color"},
                "r01_ex531": {"fluorophore": "r01_Cy3", "excitation_nm": 531, "emission_nm": 550, "role": "layer"},
                "r01_ex485": {"fluorophore": "r01_FAM", "excitation_nm": 485, "emission_nm": 520, "role": "color"},
            },
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "clip_negative": True,
        },
        "preprocessing": {
            "gaussian_noise_scale": 0.01,
        },
        "feature_extraction": {
            "feature_types": ["ratios", "projections", "intensity_features"],
            "include_g_channel": True,
        },
        "classification": {
            "method": "codebook_gmm",
            "codebook_gmm": {
                "covariance_type": "diag",
                "max_iter": 100,
                "tol": 1e-3,
                "n_init": 1,
                "random_state": 42,
            },
        },
        "codebook": {
            "path": "codebooks/prism30.csv",
        },
        "prism_panel": {
            "type": "PRISM30",
            "num_per_layer": 15,
            "g_layer_num": 2,
            "total_components": 30,
            "channel_grading": {
                "color_channels": 5,
                "layer_channel": 2,
            },
        },
        "evaluation": {
            "visualization": {
                "figure_size": (10, 8),
                "dpi": 300,
            }
        },
    }


if __name__ == "__main__":
    # Copy this file to the readout directory for reproducibility
    current_file_path = os.path.abspath(__file__)
    target_file_path = read_dir / os.path.basename(current_file_path)
    try:
        shutil.copy(current_file_path, target_file_path)
    except Exception as e:
        logger.warning(f"Could not copy script to readout directory: {e}")

    config_file = CONFIG_FILE if CONFIG_FILE.exists() else None
    main(
        intensity_file=INTENSITY_FILE,
        position_file=POSITION_FILE,
        mapping_file=MAPPING_FILE,
        corrected_file=CORRECTED_FILE,
        config_file=config_file,
    )
