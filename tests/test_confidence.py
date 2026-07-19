"""Tests for posterior-confidence utilities used in gene-calling QC filtering."""
import numpy as np
import pandas as pd
import pytest

from prism.gene_calling.confidence import top1_confidence, confident_mask
from prism.gene_calling.base import ClassificationResult


def test_top1_confidence_returns_row_max():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(top1_confidence(proba), [0.7, 0.8])


def test_top1_confidence_none_passthrough():
    assert top1_confidence(None) is None


def test_confident_mask_threshold_inclusive():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]])
    np.testing.assert_array_equal(confident_mask(proba, 0.8), [True, True, False])


def test_to_dataframe_adds_is_confident_when_threshold_given():
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.51, 0.49]])
    res = ClassificationResult(labels=np.array([1, 1, 1]), probabilities=proba)
    df = res.to_dataframe(pd.DataFrame(index=range(3)), confidence_threshold=0.8)
    assert "prediction_confidence" in df.columns
    assert "is_confident" in df.columns
    np.testing.assert_allclose(df["prediction_confidence"], [0.9, 0.6, 0.51])
    np.testing.assert_array_equal(df["is_confident"], [True, False, False])


def test_to_dataframe_no_is_confident_without_threshold():
    proba = np.array([[0.9, 0.1]])
    res = ClassificationResult(labels=np.array([1]), probabilities=proba)
    df = res.to_dataframe(pd.DataFrame(index=range(1)))
    assert "is_confident" not in df.columns


def test_to_dataframe_is_confident_false_when_no_probabilities():
    res = ClassificationResult(labels=np.array([1, 2]), probabilities=None)
    df = res.to_dataframe(pd.DataFrame(index=range(2)), confidence_threshold=0.8)
    # No probabilities -> confidence is undefined -> not confident
    assert "is_confident" in df.columns
    np.testing.assert_array_equal(df["is_confident"], [False, False])
