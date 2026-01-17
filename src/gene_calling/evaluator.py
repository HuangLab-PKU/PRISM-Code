"""
Evaluation and visualization tools for signal classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy import stats
from tqdm import tqdm
from .base import ClassificationResult


class ClassificationEvaluator:
    """
    Comprehensive evaluation tools for signal classification results.
    
    Provides methods for:
    - Classification quality assessment
    - Cluster visualization
    - Statistical analysis
    - Performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visualization_config = config.get("visualization", {})
        self.evaluation_config = config.get("evaluation", {})
        self.logger = logging.getLogger(__name__)

    def plot_ratio_overview(
        self,
        data: pd.DataFrame,
        sample: int = 10000,
        bins: int = 50,
        ax=None,
        save: bool = False,
        save_quality: str = "low",
        out_path: str = "./ratio_overview.png",
    ) -> Optional[Path]:
        """
        Plot overview histograms of channel ratio distributions using standard chn naming.

        Shows histograms for ch1/A, ch2/A, ch3/A, ch4/A in four rows.
        """
        import matplotlib.pyplot as plt

        df = data.copy()
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=42)

        if ax is not None:
            ax_tmp = ax
            fig = None
        else:
            fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 5))
            ax_tmp = ax

        ax_tmp[0].hist(bins=bins, x=df["ch1/A"])
        ax_tmp[1].hist(bins=bins, x=df["ch2/A"])
        ax_tmp[2].hist(bins=bins, x=df["ch3/A"])
        ax_tmp[3].hist(bins=bins, x=df["ch4/A"])
        plt.tight_layout()

        saved_path = None
        if save:
            dpi = 300 if save_quality == "high" else None
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            saved_path = Path(out_path)
            plt.close()
        else:
            if fig is not None:
                plt.show()

        return saved_path
        
    def evaluate_classification(
        self,
                              result: ClassificationResult,
                              data: pd.DataFrame,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of classification results.
        
        Args:
            result: ClassificationResult object
            data: Original data with features
            ground_truth: Optional true labels for supervised evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        evaluation = {}
        
        # Basic statistics
        evaluation["basic_stats"] = self._calculate_basic_stats(result, data)
        
        # Cluster quality metrics
        evaluation["cluster_quality"] = self._calculate_cluster_quality(result, data)
        
        # Supervised metrics (if ground truth available)
        if ground_truth is not None:
            evaluation["supervised_metrics"] = self._calculate_supervised_metrics(
                result.labels, ground_truth
            )
            
        # Feature analysis
        evaluation["feature_analysis"] = self._analyze_features(result, data)
        
        return evaluation
    
    def _calculate_basic_stats(
        self, result: ClassificationResult, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate basic classification statistics."""
        unique_labels = np.unique(result.labels)
        n_clusters = len(unique_labels)
        n_samples = len(result.labels)
        
        # Cluster sizes
        cluster_sizes = [np.sum(result.labels == label) for label in unique_labels]
        
        stats_dict = {
            "n_clusters": n_clusters,
            "n_samples": n_samples,
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "mean_cluster_size": np.mean(cluster_sizes),
            "std_cluster_size": np.std(cluster_sizes),
        }
        
        # Confidence statistics (if probabilities available)
        if result.probabilities is not None:
            max_probs = np.max(result.probabilities, axis=1)
            stats_dict.update(
                {
                    "mean_confidence": np.mean(max_probs),
                    "std_confidence": np.std(max_probs),
                    "min_confidence": np.min(max_probs),
                    "max_confidence": np.max(max_probs),
                }
            )
            
        return stats_dict
    
    def _calculate_cluster_quality(
        self, result: ClassificationResult, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate cluster quality metrics."""
        quality = {}
        
        # For large datasets, skip expensive metrics or use sampling
        if len(data) > 10000:
            # Sample data for expensive metrics
            sample_size = min(5000, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data.iloc[sample_indices].values
            sample_labels = result.labels[sample_indices]

            logger.info(
                f"Using sampling for quality metrics: {sample_size} samples from {len(data)} total"
            )
        else:
            sample_data = data.values
            sample_labels = result.labels
        
        # Silhouette score (if we have feature data)
        if hasattr(data, "values") and data.shape[1] > 1:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            quality["silhouette_score"] = silhouette_score(sample_data, sample_labels)
            quality["calinski_harabasz_score"] = calinski_harabasz_score(sample_data, sample_labels)
            quality["davies_bouldin_score"] = davies_bouldin_score(sample_data, sample_labels)
            
        return quality
    
    def _calculate_supervised_metrics(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate supervised evaluation metrics."""
        return {
            "adjusted_rand_score": adjusted_rand_score(ground_truth, predicted),
            "normalized_mutual_info": normalized_mutual_info_score(
                ground_truth, predicted
            ),
        }

    def _analyze_features(
        self, result: ClassificationResult, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze feature distributions across clusters."""
        analysis = {}
        
        # Get numeric columns for analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                cluster_means = []
                cluster_stds = []
                
                unique_labels = np.unique(result.labels)
                for label in unique_labels:
                    mask = result.labels == label
                    if np.sum(mask) > 0:
                        cluster_means.append(np.mean(data.loc[mask, col]))
                        cluster_stds.append(np.std(data.loc[mask, col]))
                    else:
                        cluster_means.append(np.nan)
                        cluster_stds.append(np.nan)
                        
                analysis[col] = {
                    "cluster_means": cluster_means,
                    "cluster_stds": cluster_stds,
                    "overall_mean": np.mean(data[col]),
                    "overall_std": np.std(data[col]),
                }
                
        return analysis
    
    def visualize_classification(
        self,
                               result: ClassificationResult,
                               data: pd.DataFrame,
                               output_dir: Path,
        prefix: str = "classification",
    ) -> Dict[str, Path]:
        """
        Generate comprehensive visualizations of classification results.
        
        Args:
            result: ClassificationResult object
            data: Original data with features
            output_dir: Directory to save plots
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # 1. Cluster size distribution
        plot_paths["cluster_sizes"] = self._plot_cluster_sizes(
            result, output_dir / f"{prefix}_cluster_sizes.png"
        )
        
        # 2. Feature space visualization
        if "color_dim_1" in data.columns and "color_dim_2" in data.columns:
            plot_paths["feature_space"] = self._plot_feature_space(
                result, data, output_dir / f"{prefix}_feature_space.png"
            )
            
        # 3. Channel ratio distributions
        ratio_cols = ["ch1/A", "ch2/A", "ch3/A", "ch4/A"]
        available_ratios = [col for col in ratio_cols if col in data.columns]
        if available_ratios:
            plot_paths["ratio_distributions"] = self._plot_ratio_distributions(
                result,
                data,
                available_ratios,
                output_dir / f"{prefix}_ratio_distributions.png",
            )
            
        # 4. Confidence distribution
        if result.probabilities is not None:
            plot_paths["confidence_distribution"] = self._plot_confidence_distribution(
                result, output_dir / f"{prefix}_confidence_distribution.png"
            )
            
        # 5. Cluster centroids
        if result.centroids is not None:
            plot_paths["centroids"] = self._plot_centroids(
                result, data, output_dir / f"{prefix}_centroids.png"
            )
            
        return plot_paths
    
    def visualize_channel_space(
        self,
        result: ClassificationResult,
        data: pd.DataFrame,
        output_dir: Path,
        g_layer_num: int = 2,
        method=None,
    ) -> Path:
        """
        Generate comprehensive channel space visualization (merged from pipeline).

        Args:
            result: ClassificationResult object
            data: Original data with features
            output_dir: Directory to save plots
            g_layer_num: Number of G layers
            method: GMMMethod instance for preprocessing

        Returns:
            Path to saved plot
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess data using the same method as classification
        if method is not None:
            processed_data = method.preprocess(data)
        else:
            processed_data = self._preprocess_for_visualization(data)

        # Get unique labels and create color mapping
        unique_labels = np.unique(result.labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove unlabeled

        # Create color mapping
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))

        # Determine layer panels (decide mode now; masks computed after sampling)
        faux_layer_by_label = (
            method is not None
            and getattr(method, "use_layers", True) is False
            and getattr(method, "use_4d_ratios_only", False) is True
        )
        layer_masks: List[np.ndarray] = []
        layer_titles: List[str] = []
        g_layers_to_plot = 2 if faux_layer_by_label else g_layer_num

        # Create figure: square subplots, compact horizontal spacing, no legend
        n_rows = 2
        n_cols = 2 + g_layers_to_plot
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
        plt.subplots_adjust(wspace=0.15, hspace=0.2)

        # Plot parameters - automatically adjust for dataset size and use sampling for large datasets
        n_points = len(processed_data)
        max_points_for_plot = 200000  # Maximum points to plot for performance

        if n_points > max_points_for_plot:
            # Sample data for visualization
            sample_indices = np.random.choice(
                n_points, max_points_for_plot, replace=False
            )
            plot_data = processed_data.iloc[sample_indices].copy()
            plot_labels = result.labels[sample_indices]
            logger.info(
                f"Sampling {max_points_for_plot} points from {n_points} for visualization"
            )
        else:
            plot_data = processed_data.copy()
            plot_labels = result.labels

        # Now that plot_labels is defined, compute layer masks if needed
        if faux_layer_by_label:
            mask0 = (plot_labels >= 1) & (plot_labels <= 15)
            mask1 = (plot_labels >= 16) & (plot_labels <= 30)
            layer_masks = [mask0, mask1]
            layer_titles = ["Label 1-15", "Label 16-30"]

        if n_points > 100000:
            s = 0.05  # smaller points for very large datasets
            alpha = 0.15
        elif n_points > 10000:
            s = 0.2  # smaller for medium datasets
            alpha = 0.25
        else:
            s = 0.6  # smaller for small datasets
            alpha = 0.4

        # Define fixed ranges (from legacy code)
        xrange = [-0.8, 0.8]
        yrange = [-0.6, 0.8]

        # Plot 1: color_dim_1 vs color_dim_2 (main scatter plot)
        ax_scatter = ax[0, 0]
        for label in unique_labels:
            mask = plot_labels == label
            if mask.any():
                ax_scatter.scatter(
                    plot_data.loc[mask, "color_dim_1"],
                    plot_data.loc[mask, "color_dim_2"],
                    c=[label_to_color[label]],
                    s=s,
                    alpha=alpha,
                    label=f"Cluster {label}",
                )

        # Add unlabeled points in gray
        unlabeled_mask = (plot_labels == -1)
        if unlabeled_mask.any():
            ax_scatter.scatter(
                plot_data.loc[unlabeled_mask, "color_dim_1"],
                plot_data.loc[unlabeled_mask, "color_dim_2"],
                c="gray",
                s=s,
                alpha=alpha,
                label="Unlabeled",
            )

        ax_scatter.set_xlim(xrange)
        ax_scatter.set_ylim(yrange)
        ax_scatter.set_title("Channel Space - All Points")

        # Plot 2: color_dim_1 vs ch3/A (layering channel)
        ax_scatter = ax[1, 0]
        for label in unique_labels:
            mask = plot_labels == label
            if mask.any():
                ax_scatter.scatter(
                    plot_data.loc[mask, "color_dim_1"],
                    plot_data.loc[mask, "ch3/A"],
                    c=[label_to_color[label]],
                    s=s,
                    alpha=alpha,
                )

        if unlabeled_mask.any():
            ax_scatter.scatter(
                plot_data.loc[unlabeled_mask, "color_dim_1"],
                plot_data.loc[unlabeled_mask, "ch3/A"],
                c="gray",
                s=s,
                alpha=alpha,
            )

        ax_scatter.set_xlim(xrange)  # Fixed x limits for color_dim_1
        ax_scatter.set_ylim([-0.2, 2.0])  # Fixed range for ch3/A
        ax_scatter.set_title("color_dim_1 vs ch3/A (Layering Channel)")

        # Plot 3: color_dim_2 vs ch3/A (layering channel)
        ax_scatter = ax[0, 1]
        for label in unique_labels:
            mask = plot_labels == label
            if mask.any():
                ax_scatter.scatter(
                    plot_data.loc[mask, "color_dim_2"],
                    plot_data.loc[mask, "ch3/A"],
                    c=[label_to_color[label]],
                    s=s,
                    alpha=alpha,
                )

        if unlabeled_mask.any():
            ax_scatter.scatter(
                plot_data.loc[unlabeled_mask, "color_dim_2"],
                plot_data.loc[unlabeled_mask, "ch3/A"],
                c="gray",
                s=s,
                alpha=alpha,
            )

        ax_scatter.set_xlim(yrange)  # Fixed x limits for color_dim_2
        ax_scatter.set_ylim([-0.2, 2.0])  # Fixed range for ch3/A
        ax_scatter.set_title("color_dim_2 vs ch3/A (Layering Channel)")

        # Prepare G_layer array aligned with plot_data if using true G_layer panels
        plot_g_layer = plot_data.get("G_layer", np.zeros(len(plot_data), dtype=int))

        # Plot layer panels (G-layer or faux label layers)
        for layer in range(g_layers_to_plot):
            ax_scatter = ax[0, 2 + layer]
            ax_hist = ax[1, 2 + layer]

            if faux_layer_by_label:
                layer_mask = layer_masks[layer]
                layer_data = plot_data.loc[layer_mask]
                layer_labels = plot_labels[layer_mask]
            else:
                layer_mask = (plot_g_layer == layer)
                layer_data = plot_data.loc[layer_mask]
                layer_labels = plot_labels[layer_mask]

            if faux_layer_by_label:
                ax_scatter.set_title(layer_titles[layer])
            else:
                ax_scatter.set_title(f"G Layer {layer}")

            # Plot labeled points
            for label in unique_labels:
                label_mask = layer_labels == label
                if label_mask.any():
                    ax_scatter.scatter(
                        layer_data.loc[label_mask, "color_dim_1"],
                        layer_data.loc[label_mask, "color_dim_2"],
                        c=[label_to_color[label]],
                        s=s,
                        alpha=alpha,
                    )

            # Plot unlabeled points
            unlabeled_layer_mask = (layer_labels == -1)
            if unlabeled_layer_mask.any():
                ax_scatter.scatter(
                    layer_data.loc[unlabeled_layer_mask, "color_dim_1"],
                    layer_data.loc[unlabeled_layer_mask, "color_dim_2"],
                    c="gray",
                    s=s,
                    alpha=alpha,
                )

            ax_scatter.set_xlim(xrange)
            ax_scatter.set_ylim(yrange)

            # Create 2D histogram with higher resolution
            if len(layer_data) > 0:
                x, y = layer_data["color_dim_1"], layer_data["color_dim_2"]
                bins = 200  # Higher resolution for better detail
                hist, xedges, yedges = np.histogram2d(
                    x, y, bins=bins, range=[xrange, yrange]
                )
                percentile = np.percentile(hist, 95)  # Use 95th percentile as vmax
                ax_hist.hist2d(
                    x,
                    y,
                    bins=bins,
                    vmax=percentile,
                    range=[xrange, yrange],
                    cmap="inferno",
                )
                ax_hist.set_xlim(xrange)
                ax_hist.set_ylim(yrange)

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / "channel_space_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def _preprocess_for_visualization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for visualization (extract required features)."""
        processed_data = data.copy()

        # Ensure required columns exist
        required_cols = ["color_dim_1", "color_dim_2", "ch3/A", "G_layer"]
        for col in required_cols:
            if col not in processed_data.columns:
                # If columns don't exist, create dummy data
                if col == "color_dim_1":
                    processed_data[col] = np.random.normal(0, 0.1, len(processed_data))
                elif col == "color_dim_2":
                    processed_data[col] = np.random.normal(0, 0.1, len(processed_data))
                elif col == "ch3/A":
                    processed_data[col] = np.random.uniform(0, 1, len(processed_data))
                elif col == "G_layer":
                    processed_data[col] = np.random.randint(0, 2, len(processed_data))

        return processed_data

    def _plot_cluster_sizes(
        self, result: ClassificationResult, output_path: Path
    ) -> Path:
        """Plot cluster size distribution."""
        unique_labels = np.unique(result.labels)
        cluster_sizes = [np.sum(result.labels == label) for label in unique_labels]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(unique_labels)), cluster_sizes)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Points")
        plt.title("Cluster Size Distribution")
        plt.xticks(range(len(unique_labels)), unique_labels)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_feature_space(
        self, result: ClassificationResult, data: pd.DataFrame, output_path: Path
    ) -> Path:
        """Plot 2D feature space with cluster colors."""
        plt.figure(figsize=(10, 8))
        
        # Create color map
        unique_labels = np.unique(result.labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = result.labels == label
            plt.scatter(
                data.loc[mask, "color_dim_1"],
                data.loc[mask, "color_dim_2"],
                c=[colors[i]],
                label=f"Cluster {label}",
                alpha=0.6,
                s=1,
            )

        plt.xlabel("color_dim_1")
        plt.ylabel("color_dim_2")
        plt.title("Feature Space Visualization")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_ratio_distributions(
        self,
        result: ClassificationResult,
        data: pd.DataFrame,
        ratio_cols: List[str],
        output_path: Path,
    ) -> Path:
        """Plot channel ratio distributions by cluster."""
        n_cols = len(ratio_cols)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        unique_labels = np.unique(result.labels)
        
        for i, col in enumerate(ratio_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            for label in unique_labels:
                mask = result.labels == label
                if np.sum(mask) > 0:
                    ax.hist(
                        data.loc[mask, col],
                        alpha=0.5,
                        label=f"Cluster {label}",
                        bins=50,
                    )
                    
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {col}")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_confidence_distribution(
        self, result: ClassificationResult, output_path: Path
    ) -> Path:
        """Plot prediction confidence distribution."""
        max_probs = np.max(result.probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Confidence")
        plt.axvline(
            np.mean(max_probs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(max_probs):.3f}",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_centroids(
        self, result: ClassificationResult, data: pd.DataFrame, output_path: Path
    ) -> Path:
        """Plot cluster centroids in feature space."""
        if "color_dim_1" not in data.columns or "color_dim_2" not in data.columns:
            return None
            
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(
            data["color_dim_1"],
            data["color_dim_2"],
            c=result.labels,
            cmap="tab20",
            alpha=0.3,
            s=1,
        )
        
        # Plot centroids
        if isinstance(result.centroids, dict):
            # Multi-layer centroids
            for layer, centroids in result.centroids.items():
                if centroids.shape[1] >= 2:
                    plt.scatter(
                        centroids[:, 0],
                        centroids[:, 1],
                        c="red",
                        marker="x",
                        s=100,
                        linewidths=3,
                    )
        else:
            # Single layer centroids
            if result.centroids.shape[1] >= 2:
                plt.scatter(
                    result.centroids[:, 0],
                    result.centroids[:, 1],
                    c="red",
                    marker="x",
                    s=100,
                    linewidths=3,
                )

        plt.xlabel("color_dim_1")
        plt.ylabel("color_dim_2")
        plt.title("Cluster Centroids")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def generate_report(self, evaluation: Dict[str, Any], output_path: Path) -> Path:
        """Generate a comprehensive evaluation report."""
        output_path = Path(output_path)
        
        with open(output_path, "w") as f:
            f.write("# Signal Classification Evaluation Report\n\n")
            
            # Basic statistics
            f.write("## Basic Statistics\n")
            basic_stats = evaluation.get("basic_stats", {})
            for key, value in basic_stats.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Cluster quality
            f.write("## Cluster Quality Metrics\n")
            cluster_quality = evaluation.get("cluster_quality", {})
            for key, value in cluster_quality.items():
                if value is not None:
                    f.write(f"- **{key}**: {value:.4f}\n")
            f.write("\n")
            
            # Supervised metrics
            if "supervised_metrics" in evaluation:
                f.write("## Supervised Metrics\n")
                supervised_metrics = evaluation["supervised_metrics"]
                for key, value in supervised_metrics.items():
                    f.write(f"- **{key}**: {value:.4f}\n")
                f.write("\n")
                
        return output_path


class QuantitativeEvaluator:
    """
    Quantitative evaluation functionality from legacy code.

    Provides methods for calculating CDFs, accuracy metrics, and quality control.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def calculate_cdf(
        self,
        intensity: pd.DataFrame,
        start_label: int = 0,
        num_per_layer: int = 15,
        channels: List[str] = ["ch1/A", "ch2/A", "ch3/A", "ch4/A"],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate cumulative distribution functions for each cluster.

        Args:
            intensity: Input data with labels
            start_label: Starting label index
            num_per_layer: Number of clusters per layer
            channels: Channel names to use for CDF calculation

        Returns:
            Tuple of (cdfs_df, centroids)
        """
        centroids = []
        cdfs_df = pd.DataFrame()

        for i in tqdm(
            range(start_label + 1, start_label + num_per_layer + 1), desc="component"
        ):
            data_cdf = intensity[channels]
            data = intensity[intensity["label"] == i]
            data = data[channels]
            points = np.array(data)

            if len(points) == 0:
                continue

            # Calculate the mean
            mean = np.mean(points, axis=0)

            # Calculate the covariance matrix
            cov = np.cov(points, rowvar=False)

            # Calculate CDF
            m_dist_x = (data_cdf - mean) @ np.linalg.pinv(cov)
            m_dist_x = np.einsum("ij,ji->i", m_dist_x, (data_cdf - mean).T)
            probability = 1 - stats.chi2.cdf(np.array(m_dist_x), len(channels))
            cdfs_df[i] = probability
            centroids.append(mean)

        centroids = np.array(centroids)
        cdfs_df.index = intensity.index

        return cdfs_df, centroids

    def plot_mean_accuracy(
        self,
        cdfs_df: pd.DataFrame,
        intensity: pd.DataFrame,
        sample: int = 100,
        y_line: float = 0.9,
        total_num: int = 30,
        ax=None,
    ) -> Tuple[float, List[float], List[float]]:
        """
        Plot mean accuracy curve.

        Args:
            cdfs_df: CDF DataFrame
            intensity: Input data
            sample: Number of samples for plotting
            y_line: Y-axis line for accuracy threshold
            total_num: Total number of clusters
            ax: Matplotlib axes (optional)

        Returns:
            Tuple of (accuracy, x_intercepts, y_intercepts)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Sample data for plotting
        sample_indices = np.random.choice(
            len(cdfs_df), min(sample, len(cdfs_df)), replace=False
        )
        cdfs_sample = cdfs_df.iloc[sample_indices]

        # Calculate mean accuracy
        mean_accuracy = []
        x_values = np.linspace(0, 1, 100)

        for x in x_values:
            accuracy = np.mean(cdfs_sample > x)
            mean_accuracy.append(accuracy)

        # Plot
        ax.plot(x_values, mean_accuracy, "b-", linewidth=2, label="Mean Accuracy")
        ax.axhline(y=y_line, color="r", linestyle="--", label=f"Threshold: {y_line}")
        ax.set_xlabel("Probability Threshold")
        ax.set_ylabel("Mean Accuracy")
        ax.set_title("Mean Accuracy vs Probability Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Find intercepts
        y_intercepts = np.array(mean_accuracy)
        x_intercepts = x_values

        # Find where accuracy crosses the threshold
        threshold_idx = np.where(y_intercepts >= y_line)[0]
        if len(threshold_idx) > 0:
            accuracy = x_intercepts[threshold_idx[0]]
        else:
            accuracy = 1.0

        return accuracy, x_intercepts.tolist(), y_intercepts.tolist()

    def count_distribution(
        self,
        intensity: pd.DataFrame,
        num_per_layer: int = 15,
        G_layer: int = 2,
        out_path: Optional[str] = None,
    ) -> None:
        """
        Plot cluster size distribution.

        Args:
            intensity: Input data with labels
            num_per_layer: Number of clusters per layer
            G_layer: Number of G layers
            out_path: Output path for plot (optional)
        """
        plt.figure(figsize=(num_per_layer * G_layer / 7, 3))
        sns.barplot(
            x=[cluster_num + 1 for cluster_num in range(num_per_layer * G_layer)],
            y=[
                len(intensity[intensity["label"] == cluster_num + 1])
                for cluster_num in range(num_per_layer * G_layer)
            ],
        )
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Points")
        plt.title("Cluster Size Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def calculate_overlap_matrix(
        self,
        cdfs_df: pd.DataFrame,
        intensity: pd.DataFrame,
        p_thre: float = 0.1,
        total_num: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate overlap matrix between clusters.

        Args:
            cdfs_df: CDF DataFrame
            intensity: Input data with labels
            p_thre: Probability threshold
            total_num: Total number of clusters

        Returns:
            Overlap matrix DataFrame
        """
        overlap = pd.DataFrame()

        for cluster_num in range(1, total_num + 1):
            tmp = cdfs_df.loc[
                intensity["label"][intensity["label"] == cluster_num].index
            ]
            if len(tmp) > 0:
                overlap[cluster_num] = (tmp > p_thre).sum(axis=0) / len(tmp)

        return overlap

    def evaluate_classification_quality(
        self,
        intensity: pd.DataFrame,
        num_per_layer: int = 15,
        G_layer: int = 2,
        channels: List[str] = ["ch1/A", "ch2/A", "ch3/A", "ch4/A"],
        p_thre_list: List[float] = [0.1, 0.5],
    ) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation of classification results.

        Args:
            intensity: Input data with labels
            num_per_layer: Number of clusters per layer
            G_layer: Number of G layers
            channels: Channels for evaluation
            p_thre_list: List of probability thresholds

        Returns:
            Dictionary containing evaluation results
        """
        results = {}

        # Calculate CDFs
        cdfs_df, centroids = self.calculate_cdf(
            intensity, st=0, num_per_layer=G_layer * num_per_layer, channel=channels
        )
        results["cdfs_df"] = cdfs_df
        results["centroids"] = centroids

        # Calculate overlap matrices
        overlap_matrices = {}
        for p_thre in p_thre_list:
            overlap = self.calculate_overlap_matrix(
                cdfs_df, intensity, p_thre=p_thre, total_num=G_layer * num_per_layer
            )
            overlap_matrices[p_thre] = overlap
        results["overlap_matrices"] = overlap_matrices

        # Calculate mean accuracy
        accuracy, x_intercepts, y_intercepts = self.plot_mean_accuracy(
            cdfs_df, intensity, total_num=G_layer * num_per_layer
        )
        results["mean_accuracy"] = accuracy
        results["accuracy_curve"] = {"x": x_intercepts, "y": y_intercepts}

        return results
