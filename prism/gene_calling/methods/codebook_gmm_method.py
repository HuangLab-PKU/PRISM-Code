"""
Codebook-initialized GMM for gene calling.

Uses a codebook CSV to define gene encodings and initialize GMM centroids.
Operates as a single GMM over all genes (no explicit layer separation).
Produces globally normalized probabilities via log-likelihood + softmax.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from ..base import ClassificationResult

logger = logging.getLogger(__name__)


class CodebookGMMMethod:
    """Codebook-initialized GMM for gene calling.

    Key differences from the legacy ``GMMMethod``:

    * Initialization from codebook rather than empirical maxima finding.
    * A single GMM over all genes (no layer-split fitting).
    * Globally normalized probabilities (log-likelihood + softmax).
    * Soft classification output (top-1 and top-2 with probabilities).
    * No hardcoded PRISM panel types.

    Parameters
    ----------
    config : dict
        Method-specific parameters.  Recognized keys::

            covariance_type : str   ("diag")
            max_iter : int          (100)
            tol : float             (1e-3)
            n_init : int            (1)
            random_state : int      (42)
            gaussian_noise_scale : float (0.01)
            layer_log_transform : bool   (True)
            centroid_max_movement : float (0.15)
            constrained_fit_rounds : int  (1)

    codebook_df : pd.DataFrame
        Must have a ``gene`` column plus one column per fluorophore.
        Values are integer grading levels.
    channel_roles : dict
        ``{fluorophore_name: "color" | "layer", ...}``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        codebook_df: pd.DataFrame,
        channel_roles: Dict[str, str],
    ):
        self.config = config
        self.codebook = codebook_df.copy()
        self.n_genes = len(codebook_df)
        self.channel_roles = channel_roles

        # Separate color and layer channels (preserving codebook column order)
        fluorophore_cols = [c for c in codebook_df.columns if c != "gene"]
        self.color_channels: List[str] = [
            c for c in fluorophore_cols if channel_roles.get(c) == "color"
        ]
        self.layer_channels: List[str] = [
            c for c in fluorophore_cols if channel_roles.get(c) == "layer"
        ]
        self.fluorophore_cols = fluorophore_cols

        # GMM hyper-parameters
        self.covariance_type = config.get("covariance_type", "diag")
        self.max_iter = config.get("max_iter", 100)
        self.tol = config.get("tol", 1e-3)
        self.n_init = config.get("n_init", 1)
        self.random_state = config.get("random_state", 42)
        self.gaussian_noise_scale = config.get("gaussian_noise_scale", 0.01)
        self.layer_log_transform = config.get("layer_log_transform", True)

        # Centroid constraint
        self.centroid_max_movement = config.get("centroid_max_movement", 0.15)
        self.constrained_fit_rounds = config.get("constrained_fit_rounds", 1)

        # State
        self.model: Optional[GaussianMixture] = None
        self.is_fitted = False
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert corrected fluorophore intensities to ratio features.

        For color channels the ratio is ``intensity / color_sum``.
        For layer channels the ratio is ``intensity / color_sum`` optionally
        followed by a log transform.

        Returns a DataFrame whose columns are the feature names.
        """
        color_sum = data[self.color_channels].sum(axis=1).replace(0, np.nan)

        features: Dict[str, np.ndarray] = {}
        for ch in self.color_channels:
            features[f"{ch}_ratio"] = (data[ch] / color_sum).values

        for ch in self.layer_channels:
            ratio = (data[ch] / color_sum).values
            if self.layer_log_transform:
                ratio = np.log1p(ratio)
            features[f"{ch}_ratio"] = ratio

        feat_df = pd.DataFrame(features, index=data.index)

        # Add small Gaussian noise to avoid singular covariance at boundaries
        if self.gaussian_noise_scale > 0:
            rng = np.random.default_rng(self.random_state)
            noise = rng.normal(0, self.gaussian_noise_scale, size=feat_df.shape)
            feat_df = feat_df + noise

        return feat_df

    def _codebook_to_centroids(self, feature_names: List[str]) -> np.ndarray:
        """Convert codebook grading levels to ratio-space centroids.

        Color channels: ``grade / max_color_grade`` where
        ``max_color_grade = sum of color grades for that gene``.

        Layer channels: same ratio scheme, then ``log1p`` if enabled.

        Returns array of shape ``(n_genes, n_features)``.
        """
        centroids = np.zeros((self.n_genes, len(feature_names)))

        for i in range(self.n_genes):
            row = self.codebook.iloc[i]
            color_sum = sum(row[c] for c in self.color_channels)
            if color_sum == 0:
                color_sum = 1  # avoid division by zero for edge-case genes

            for j, feat_name in enumerate(feature_names):
                # feat_name is e.g. "r01_Cy5_ratio"
                fluor = feat_name.replace("_ratio", "")
                grade = row.get(fluor, 0)

                ratio = grade / color_sum

                if fluor in self.layer_channels and self.layer_log_transform:
                    ratio = np.log1p(ratio)

                centroids[i, j] = ratio

        return centroids

    # ------------------------------------------------------------------
    # Public API (matches GMMMethod interface used by pipeline)
    # ------------------------------------------------------------------

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """No-op: correction is done at pipeline level, feature extraction
        is internal to fit/predict.  Return data unchanged."""
        return data

    def fit(
        self,
        data: pd.DataFrame,
        ground_truth: Optional[np.ndarray] = None,
        metadata: Optional[Any] = None,
    ) -> "CodebookGMMMethod":
        """Fit the codebook-initialized GMM.

        Parameters
        ----------
        data : pd.DataFrame
            Corrected intensity data with fluorophore-named columns.
        ground_truth : array-like, optional
            Ignored (unsupervised).
        metadata : optional
            Ignored.
        """
        logger.info(
            "Fitting CodebookGMM: %d genes, %d data points",
            self.n_genes,
            len(data),
        )

        features = self._compute_features(data)
        self._feature_names = list(features.columns)
        centroids = self._codebook_to_centroids(self._feature_names)

        logger.info(
            "Initial centroids from codebook (shape %s), features shape %s",
            centroids.shape,
            features.shape,
        )

        self.model = GaussianMixture(
            n_components=self.n_genes,
            means_init=centroids,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.model.fit(features.values)

        # Constrain centroid movement
        self._enforce_centroid_constraint(centroids)

        self.is_fitted = True
        logger.info("CodebookGMM fit complete (converged=%s)", self.model.converged_)
        return self

    def predict(self, data: pd.DataFrame) -> ClassificationResult:
        """Predict gene labels with globally normalized probabilities.

        Parameters
        ----------
        data : pd.DataFrame
            Corrected intensity data with fluorophore-named columns.

        Returns
        -------
        ClassificationResult
            With ``probabilities`` shape ``(n_samples, n_genes)`` and
            metadata containing the codebook.
        """
        if not self.is_fitted:
            raise RuntimeError("CodebookGMMMethod must be fitted before predict()")

        features = self._compute_features(data)

        # Globally normalized probabilities via log-likelihood + softmax
        log_probs = self.model._estimate_weighted_log_prob(features.values)
        log_norm = logsumexp(log_probs, axis=1, keepdims=True)
        probabilities = np.exp(log_probs - log_norm)

        labels = np.argmax(probabilities, axis=1)

        return ClassificationResult(
            labels=labels,
            probabilities=probabilities,
            centroids=self.model.means_,
            metadata={
                "codebook": self.codebook,
                "method": "CodebookGMM",
                "feature_names": self._feature_names,
            },
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic information about the fitted model."""
        info: Dict[str, Any] = {
            "method": "CodebookGMM",
            "is_fitted": self.is_fitted,
            "n_genes": self.n_genes,
            "color_channels": self.color_channels,
            "layer_channels": self.layer_channels,
            "config": self.config,
        }
        if self.is_fitted and self.model is not None:
            info["converged"] = self.model.converged_
            info["n_iter"] = self.model.n_iter_
            info["bic"] = None  # Requires data; caller can compute if needed
        return info

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Feature importance is not directly available for GMM."""
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enforce_centroid_constraint(self, initial_centroids: np.ndarray) -> None:
        """Clamp fitted centroids that drifted too far from codebook init."""
        if self.centroid_max_movement <= 0:
            return

        for _ in range(self.constrained_fit_rounds):
            displacement = np.linalg.norm(
                self.model.means_ - initial_centroids, axis=1
            )
            violated = displacement > self.centroid_max_movement
            if not np.any(violated):
                break

            n_violated = int(np.sum(violated))
            logger.info(
                "Centroid constraint: %d / %d components exceeded max movement %.3f",
                n_violated,
                self.n_genes,
                self.centroid_max_movement,
            )
            # Reset violated centroids to initial position
            self.model.means_[violated] = initial_centroids[violated]
