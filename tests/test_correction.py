"""Tests for the channel-correction step shared by all gene-calling methods."""
import numpy as np
import pandas as pd
import pytest

from prism.gene_calling.correction import apply_channel_correction


CHANNELS = {
    "r01_ex628": {"fluorophore": "r01_Cy5", "role": "color"},
    "r01_ex531": {"fluorophore": "r01_Cy3", "role": "layer"},
}


def _frame():
    return pd.DataFrame(
        {
            "index": [0, 1],
            "r01_ex628": [100.0, 200.0],
            "r01_ex531": [10.0, 20.0],
            "Y": [5, 6],
        }
    )


def test_identity_matrix_renames_without_changing_values():
    result = apply_channel_correction(_frame(), CHANNELS, np.eye(2))

    np.testing.assert_allclose(result["r01_Cy5"], [100.0, 200.0])
    np.testing.assert_allclose(result["r01_Cy3"], [10.0, 20.0])
    assert "r01_ex628" not in result.columns


def test_matrix_is_applied_as_raw_at_m_transpose():
    # Subtract 50% of Cy3 crosstalk out of the Cy5 channel.
    M = np.array([[1.0, -0.5], [0.0, 1.0]])

    result = apply_channel_correction(_frame(), CHANNELS, M, clip_negative=False)

    np.testing.assert_allclose(result["r01_Cy5"], [95.0, 190.0])
    np.testing.assert_allclose(result["r01_Cy3"], [10.0, 20.0])


def test_clip_negative_floors_at_zero():
    M = np.array([[1.0, -20.0], [0.0, 1.0]])  # drives Cy5 negative

    clipped = apply_channel_correction(_frame(), CHANNELS, M, clip_negative=True)
    unclipped = apply_channel_correction(_frame(), CHANNELS, M, clip_negative=False)

    np.testing.assert_allclose(clipped["r01_Cy5"], [0.0, 0.0])
    assert (unclipped["r01_Cy5"] < 0).all()


def test_non_channel_columns_are_preserved():
    result = apply_channel_correction(_frame(), CHANNELS, np.eye(2))

    np.testing.assert_array_equal(result["index"], [0, 1])
    np.testing.assert_array_equal(result["Y"], [5, 6])


def test_missing_channel_column_raises_keyerror_naming_it():
    data = _frame().drop(columns=["r01_ex531"])

    with pytest.raises(KeyError, match="r01_ex531"):
        apply_channel_correction(data, CHANNELS, np.eye(2))


def test_matrix_shape_mismatch_raises_valueerror():
    with pytest.raises(ValueError, match="does not match"):
        apply_channel_correction(_frame(), CHANNELS, np.eye(3))
