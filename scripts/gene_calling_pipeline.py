#!/usr/bin/env python3
"""
Gene Calling Pipeline
Main execution script for the gene calling framework
Supports multi-channel signal point classification following PRISM methodology
"""

import sys
import argparse
import pandas as pd
import yaml
import shutil
import time
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gene_calling import (
    SignalClassificationPipeline,
    ManualThresholdAdjuster,
    QuantitativeEvaluator,
    load_gene_calling_config,
)


# Setup logging
def setup_logging(output_dir: Path):
    """Setup logging to output to target directory."""
    log_file = output_dir / "gene_calling.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also output to console
        ],
    )


logger = logging.getLogger(__name__)


# Configuration loading is now handled by the gene_calling module


def copy_config_to_output(config_path: str, output_dir: Path, config: dict):
    """Copy configuration to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original config file with gene_calling prefix
    shutil.copy2(config_path, output_dir / "gene_calling_config.yaml")

    # Save processed config with gene_calling prefix
    with open(output_dir / "gene_calling_full_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Configuration copied to: {output_dir}")


def load_intensity_data(data_path: str) -> pd.DataFrame:
    """Load intensity data from CSV file"""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded intensity data: {data_path}, shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load intensity data from {data_path}: {e}")
        raise


def load_coordinates_data(coordinates_path: str) -> pd.DataFrame:
    """Load coordinates data from CSV file"""
    try:
        coordinates = pd.read_csv(coordinates_path)
        logger.info(
            f"Loaded coordinates data: {coordinates_path}, shape: {coordinates.shape}"
        )
        return coordinates
    except Exception as e:
        logger.error(f"Failed to load coordinates data from {coordinates_path}: {e}")
        raise


def process_gene_calling_pipeline(
    intensity_data: pd.DataFrame,
    coordinates_data: pd.DataFrame,
    pipeline: SignalClassificationPipeline,
    output_dir: str,
    config: dict,
) -> bool:
    """Process gene calling pipeline"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to output directory
    setup_logging(output_dir)

    # Combine data
    start_time = time.time()
    data = pd.concat([intensity_data, coordinates_data], axis=1)
    logger.info(f"Combined data shape: {data.shape}")
    combine_time = time.time() - start_time
    logger.info(f"Data combination took: {combine_time:.2f} seconds")

    # Fit pipeline (handles preprocessing and feature extraction internally)
    logger.info("Fitting pipeline...")
    fit_start_time = time.time()
    pipeline.fit(data)
    fit_time = time.time() - fit_start_time
    logger.info(f"Pipeline.fit completed successfully in {fit_time:.2f} seconds")

    # Make predictions
    logger.info("Making predictions...")
    predict_start_time = time.time()
    result = pipeline.predict(data)
    predict_time = time.time() - predict_start_time
    logger.info(f"Prediction completed in {predict_time:.2f} seconds")

    # Evaluate results
    logger.info("Evaluating results...")
    eval_start_time = time.time()
    pipeline.evaluate(result, data, None, output_dir)
    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    # Save results
    result_df = result.to_dataframe(data)
    result_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save model
    model_path = output_dir / "trained_model.yaml"
    pipeline.save_model(model_path)

    # Additional processing based on config
    if config.get("manual_threshold_adjustment", {}).get("enabled", False):
        logger.info("Applying manual threshold adjustment...")
        adjuster = ManualThresholdAdjuster(
            config.get("manual_threshold_adjustment", {})
        )
        mask_dir = config.get("manual_threshold_adjustment", {}).get("mask_dir")
        if mask_dir and Path(mask_dir).exists():
            relabeled_data = adjuster.relabel(result_df, mask_dir, mode="discard")
            relabeled_data.to_csv(output_dir / "predictions_relabeled.csv", index=False)
            logger.info("Manual threshold adjustment completed")

    if config.get("quantitative_evaluation", {}).get("enabled", False):
        logger.info("Running quantitative evaluation...")
        quant_eval = QuantitativeEvaluator(config.get("quantitative_evaluation", {}))
        quality_results = quant_eval.evaluate_classification_quality(
            result_df,
            num_per_layer=config.get("classification", {})
            .get("gmm", {})
            .get("num_per_layer", 15),
            G_layer=config.get("preprocessing", {}).get("g_layer_num", 2),
        )

        # Save quantitative evaluation results
        with open(output_dir / "quantitative_evaluation.yaml", "w") as f:
            yaml.dump(quality_results, f, default_flow_style=False)
        logger.info("Quantitative evaluation completed")

    if config.get("data_overview", {}).get("enabled", False):
        logger.info("Generating data overview...")
        evaluator.plot_ratio_overview(
            processed_data,
            sample=config.get("data_overview", {}).get("sample_size", 10000),
            bins=config.get("data_overview", {}).get("bins", 50),
            save=True,
            save_quality="high",
            out_path=str(output_dir / "data_overview.png"),
        )
        logger.info("Data overview generated")

    logger.info("Gene calling pipeline completed successfully")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Gene Calling Pipeline for PRISM signal point classification"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/gene_calling.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Path to data directory containing intensity and coordinates files",
    )
    parser.add_argument(
        "--run-id", "-r", type=str, required=True, help="Run ID for the experiment"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: data_dir/run_id/readout/classification)",
    )
    parser.add_argument(
        "--intensity-file",
        type=str,
        default="intensity.csv",
        help="Intensity data filename (default: intensity.csv)",
    )
    parser.add_argument(
        "--coordinates-file",
        type=str,
        default="coordinates.csv",
        help="Coordinates data filename (default: coordinates.csv)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_gene_calling_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    # Create pipeline
    pipeline = SignalClassificationPipeline(config=config)

    # Set up paths
    data_dir = Path(args.data_dir)
    intensity_file = data_dir / args.intensity_file
    coordinates_file = data_dir / args.coordinates_file

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Output to runid's readout directory instead of readout/classification
        output_dir = data_dir

    logger.info(f"Input directory: {data_dir}")
    logger.info(f"Intensity file: {intensity_file}")
    logger.info(f"Coordinates file: {coordinates_file}")
    logger.info(f"Output directory: {output_dir}")

    # Copy config to output directory
    copy_config_to_output(args.config, output_dir, config)

    # Load data
    intensity_data = load_intensity_data(str(intensity_file))
    coordinates_data = load_coordinates_data(str(coordinates_file))

    # Process gene calling pipeline
    success = process_gene_calling_pipeline(
        intensity_data, coordinates_data, pipeline, str(output_dir), config
    )

    if success:
        logger.info("=" * 50)
        logger.info("Gene calling pipeline completed successfully!")
        logger.info(f"  Run ID: {args.run_id}")
        logger.info(f"  Data points processed: {len(intensity_data)}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("=" * 50)
        return 0
    else:
        logger.error("Gene calling pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
