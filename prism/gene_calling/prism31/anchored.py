"""Fit a codebook-anchored isotropic Gaussian mixture in the PR colour plane."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


class AnchoredMixture:
    def __init__(self, anchors, max_shift_fraction):
        self.anchors = np.asarray(anchors, dtype=float)
        self.fraction = float(max_shift_fraction)
        if not 0 <= self.fraction < .5:
            raise ValueError("Centre displacement must be below half the neighbour spacing")
        distances = cdist(self.anchors, self.anchors)
        np.fill_diagonal(distances, np.inf)
        self.spacing = distances.min(axis=1)
        if not np.all(np.isfinite(self.spacing) & (self.spacing > 0)):
            raise ValueError("Distinct finite anchors required")

    def scores(self, x):
        squared = cdist(x, self.means, "sqeuclidean")
        return (np.log(self.weights) - x.shape[1] * np.log(self.sigma)
                - squared / (2 * self.sigma**2))

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.means = self.anchors.copy()
        self.sigma = self.spacing / 4
        self.weights = np.full(len(self.means), 1 / len(self.means))
        previous = -np.inf
        for _ in range(100):
            scores = self.scores(x)
            ll = logsumexp(scores, axis=1)
            resp = np.exp(scores - ll[:, None])
            mass = resp.sum(axis=0).clip(1e-12)
            move = resp.T @ x / mass[:, None] - self.anchors
            scale = np.minimum(1, self.fraction * self.spacing /
                               np.maximum(1e-12, np.linalg.norm(move, axis=1)))
            self.means = self.anchors + move * scale[:, None]
            squared = cdist(x, self.means, "sqeuclidean")
            self.sigma = np.clip(np.sqrt((resp * squared).sum(axis=0) /
                                         (x.shape[1] * mass)),
                                 .08 * self.spacing, .45 * self.spacing)
            self.weights = (mass + 1) / (mass.sum() + len(mass))
            if abs(ll.mean() - previous) < 1e-5:
                break
            previous = ll.mean()
        return self

    def classify(self, x):
        scores = self.scores(x)
        ll = logsumexp(scores, axis=1)
        labels = scores.argmax(axis=1)
        return labels, np.exp(scores[np.arange(len(x)), labels] - ll), ll
