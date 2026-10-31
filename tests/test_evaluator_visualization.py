"""Tests for visualization input validation.

A colourspace figure rendered from fabricated data is indistinguishable from a
real one, so a schema mismatch has to fail loudly rather than be filled in.
"""
import pandas as pd
import pytest

from prism.gene_calling.evaluator import ClassificationEvaluator


REQUIRED = ["color_dim_1", "color_dim_2", "ch3/A", "G_layer"]


def _evaluator():
    return ClassificationEvaluator({})


def test_complete_data_passes_through_unchanged():
    data = pd.DataFrame({col: [0.1, 0.2] for col in REQUIRED})

    result = _evaluator()._preprocess_for_visualization(data)

    for col in REQUIRED:
        pd.testing.assert_series_equal(result[col], data[col])


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_column_raises_naming_it(missing):
    data = pd.DataFrame({col: [0.1, 0.2] for col in REQUIRED if col != missing})

    with pytest.raises(ValueError, match=missing.replace("/", "/")):
        _evaluator()._preprocess_for_visualization(data)


def test_error_lists_every_missing_column():
    data = pd.DataFrame({"color_dim_1": [0.1, 0.2]})

    with pytest.raises(ValueError) as excinfo:
        _evaluator()._preprocess_for_visualization(data)

    message = str(excinfo.value)
    for col in ["color_dim_2", "ch3/A", "G_layer"]:
        assert col in message


def test_does_not_fabricate_data_for_missing_columns():
    """Regression: missing columns were filled with np.random, yielding a
    plausible-looking colourspace plot made of pure noise."""
    data = pd.DataFrame({"color_dim_1": [0.1, 0.2]})

    with pytest.raises(ValueError):
        _evaluator()._preprocess_for_visualization(data)
