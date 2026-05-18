"""
Estimate a channel correction (unmixing) matrix from calibration experiments.

Given pure-color spot intensity CSVs (one per fluorophore) and optionally a
mixed-color CSV (all dyes present), this script estimates the 4x4
``transform_matrix`` that removes spectral crosstalk.

Algorithm
---------
1. For each pure-dye CSV, compute the median intensity vector across all
   channels.  This yields one column of the *mixing matrix* **S**.
2. Assemble **S** (n_channels x n_dyes).
3. Compute the *unmixing matrix* **U = S^{-1}** (or pseudo-inverse when
   the matrix is ill-conditioned).
4. Optionally validate on a mixed-color dataset.
5. Write the resulting ``channel_correction.transform_matrix`` as a YAML
   snippet ready to paste into ``gene_calling_base.yaml``.

Usage
-----
::

    python calibrate_channels.py \\
        --pure-dir ./calibration/pure/ \\
        --channel-order r01_ex628 r01_ex586 r01_ex531 r01_ex485 \\
        --mixed-file ./calibration/mixed.csv \\
        --output channel_correction.yaml

Pure-color CSV files
~~~~~~~~~~~~~~~~~~~~
Each CSV must contain columns matching the ``--channel-order`` list.  Each
file represents spots from a single-fluorophore-only experiment.  The files
are matched to channels by a ``--pure-mapping`` argument (default: file
stem order matches ``--channel-order``).

Output
------
- ``channel_correction.yaml``: YAML snippet with ``transform_matrix``.
- ``mixing_matrix.png`` (optional): Heatmap of the estimated mixing matrix.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_mixing_matrix(
    pure_files: List[Path],
    channel_order: List[str],
) -> np.ndarray:
    """Build the mixing matrix S from pure-color CSVs.

    Each pure-color file contributes one column (median intensity across
    all spots in each channel).

    Parameters
    ----------
    pure_files : list of Path
        Paths to pure-color CSVs, one per dye, in the order matching
        *channel_order*.
    channel_order : list of str
        Column names in the CSVs that correspond to channels (e.g.
        ``["r01_ex628", "r01_ex586", "r01_ex531", "r01_ex485"]``).

    Returns
    -------
    np.ndarray, shape (n_channels, n_channels)
        The mixing matrix **S**.  Column *j* is the median response of
        dye *j* in all channels.
    """
    n = len(channel_order)
    if len(pure_files) != n:
        raise ValueError(
            f"Number of pure-color files ({len(pure_files)}) must match "
            f"number of channels ({n})"
        )

    S = np.zeros((n, n))
    for j, csv_path in enumerate(pure_files):
        df = pd.read_csv(csv_path)
        missing = [c for c in channel_order if c not in df.columns]
        if missing:
            raise KeyError(
                f"{csv_path.name} is missing columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        medians = df[channel_order].median().values.astype(np.float64)
        S[:, j] = medians
        logger.info("Dye %d (%s): medians = %s", j, csv_path.name, medians)

    return S


def estimate_unmixing_matrix(S: np.ndarray) -> np.ndarray:
    """Compute unmixing matrix U = S^{-1}.

    Falls back to the pseudo-inverse if S is singular or ill-conditioned.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Mixing matrix.

    Returns
    -------
    np.ndarray, shape (n, n)
        Unmixing matrix.
    """
    cond = np.linalg.cond(S)
    logger.info("Mixing matrix condition number: %.2f", cond)

    if cond > 1e6:
        logger.warning(
            "Mixing matrix is ill-conditioned (cond=%.1e). "
            "Using pseudo-inverse -- results may be unreliable.",
            cond,
        )
        U = np.linalg.pinv(S)
    else:
        U = np.linalg.inv(S)

    return U


def validate_on_mixed(
    U: np.ndarray,
    mixed_file: Path,
    channel_order: List[str],
) -> None:
    """Log validation statistics by applying U to mixed-color data.

    After correction, each spot should have most intensity in a single
    channel (or at least follow the codebook pattern).
    """
    df = pd.read_csv(mixed_file)
    raw = df[channel_order].values.astype(np.float64)
    corrected = raw @ U.T
    corrected = np.maximum(corrected, 0)

    logger.info("--- Validation on mixed-color data (%d spots) ---", len(df))
    for i, ch in enumerate(channel_order):
        col = corrected[:, i]
        logger.info(
            "  %s: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
            ch,
            col.mean(),
            col.std(),
            col.min(),
            col.max(),
        )

    # Purity metric: fraction of intensity in dominant channel per spot
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    fractions = corrected / row_sums
    dominant_fraction = fractions.max(axis=1)
    logger.info(
        "  Dominant channel fraction: mean=%.3f, median=%.3f",
        dominant_fraction.mean(),
        np.median(dominant_fraction),
    )


def matrix_to_yaml_list(M: np.ndarray, decimals: int = 6) -> list:
    """Convert a numpy matrix to a nested list suitable for YAML output."""
    return [[round(float(x), decimals) for x in row] for row in M]


def save_yaml(U: np.ndarray, output_path: Path, channel_order: List[str]) -> None:
    """Save the unmixing matrix as a YAML snippet."""
    snippet = {
        "channel_correction": {
            "transform_matrix": matrix_to_yaml_list(U),
            "_channel_order": channel_order,
            "_note": (
                "Estimated by calibrate_channels.py. "
                "Copy transform_matrix into gene_calling_base.yaml."
            ),
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(snippet, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
    logger.info("Saved unmixing matrix to %s", output_path)


def try_plot_mixing_matrix(
    S: np.ndarray,
    U: np.ndarray,
    channel_order: List[str],
    output_dir: Path,
) -> None:
    """Save a diagnostic heatmap of S and U (best-effort, skip on error)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, mat, title in [
            (axes[0], S, "Mixing matrix S"),
            (axes[1], U, "Unmixing matrix U"),
        ]:
            im = ax.imshow(mat, cmap="RdBu_r", aspect="equal")
            ax.set_title(title)
            ax.set_xticks(range(len(channel_order)))
            ax.set_yticks(range(len(channel_order)))
            ax.set_xticklabels(channel_order, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(channel_order, fontsize=8)
            for (i, j), val in np.ndenumerate(mat):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        plot_path = output_dir / "mixing_matrix.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Saved diagnostic plot to %s", plot_path)
    except ImportError:
        logger.info("matplotlib not available -- skipping diagnostic plot")
    except Exception as e:
        logger.warning("Could not generate diagnostic plot: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate channel correction (unmixing) matrix from calibration data"
    )
    parser.add_argument(
        "--pure-dir",
        type=Path,
        required=True,
        help="Directory containing pure-color CSV files (one per dye)",
    )
    parser.add_argument(
        "--channel-order",
        nargs="+",
        default=["r01_ex628", "r01_ex586", "r01_ex531", "r01_ex485"],
        help="Channel column names in order (default: r01_ex628 r01_ex586 r01_ex531 r01_ex485)",
    )
    parser.add_argument(
        "--pure-pattern",
        type=str,
        default="*.csv",
        help='Glob pattern for pure-color CSV files (default: "*.csv")',
    )
    parser.add_argument(
        "--mixed-file",
        type=Path,
        default=None,
        help="Optional mixed-color CSV for validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("channel_correction.yaml"),
        help="Output YAML file (default: channel_correction.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Discover pure-color files (sorted alphabetically to match channel order)
    pure_files = sorted(args.pure_dir.glob(args.pure_pattern))
    if len(pure_files) != len(args.channel_order):
        raise RuntimeError(
            f"Found {len(pure_files)} pure-color CSVs in {args.pure_dir} "
            f"(pattern: {args.pure_pattern}), but expected {len(args.channel_order)} "
            f"(one per channel).  Files found: {[f.name for f in pure_files]}"
        )

    logger.info(
        "Pure-color files (in channel order): %s",
        [f.name for f in pure_files],
    )

    # Step 1-2: Build mixing matrix
    S = compute_mixing_matrix(pure_files, args.channel_order)

    # Step 3: Unmixing
    U = estimate_unmixing_matrix(S)
    logger.info("Unmixing matrix U:\n%s", U)

    # Step 4: Validate
    if args.mixed_file and args.mixed_file.exists():
        validate_on_mixed(U, args.mixed_file, args.channel_order)

    # Step 5: Save
    save_yaml(U, args.output, args.channel_order)

    # Diagnostic plot
    output_dir = args.output.parent
    try_plot_mixing_matrix(S, U, args.channel_order, output_dir)

    logger.info("Calibration complete.")


if __name__ == "__main__":
    main()
