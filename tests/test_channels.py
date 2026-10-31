"""Tests for resolving stitched-image filenames to canonical channel names.

Upstream stitching decides what the TIFFs on disk are called (``cyc_1_cy5.tif``);
PRISM decides what the intensity columns are called (``r01_ex628``).  Keeping
those separate is what lets the readout -> gene_calling handoff be stable
without renaming anyone's data.
"""
import pytest

from prism.readout.channels import resolve_channels


def test_explicit_names_are_used_verbatim():
    cfg = {
        "channels": [
            {"file": "cyc_1_cy5.tif", "name": "r01_ex628"},
            {"file": "cyc_1_cy3.tif", "name": "r01_ex531"},
        ]
    }

    assert resolve_channels(cfg) == [
        ("cyc_1_cy5.tif", "r01_ex628"),
        ("cyc_1_cy3.tif", "r01_ex531"),
    ]


def test_name_defaults_to_file_stem():
    cfg = {"channels": [{"file": "cyc_1_cy5.tif"}]}

    assert resolve_channels(cfg) == [("cyc_1_cy5.tif", "cyc_1_cy5")]


def test_legacy_channel_files_key_still_works():
    """Existing configs use a bare filename list; those must keep working."""
    cfg = {"channel_files": ["cyc_1_cy5.tif", "cyc_1_cy3.tif"]}

    assert resolve_channels(cfg) == [
        ("cyc_1_cy5.tif", "cyc_1_cy5"),
        ("cyc_1_cy3.tif", "cyc_1_cy3"),
    ]


def test_channels_takes_precedence_over_channel_files():
    cfg = {
        "channels": [{"file": "a.tif", "name": "chA"}],
        "channel_files": ["b.tif"],
    }

    assert resolve_channels(cfg) == [("a.tif", "chA")]


def test_missing_channel_declaration_raises():
    with pytest.raises(KeyError, match="channels"):
        resolve_channels({})


def test_empty_channel_list_raises():
    with pytest.raises(ValueError, match="at least one channel"):
        resolve_channels({"channels": []})


def test_duplicate_channel_names_raise():
    cfg = {
        "channels": [
            {"file": "a.tif", "name": "dup"},
            {"file": "b.tif", "name": "dup"},
        ]
    }

    with pytest.raises(ValueError, match="dup"):
        resolve_channels(cfg)


def test_channel_entry_without_file_raises():
    with pytest.raises(KeyError, match="file"):
        resolve_channels({"channels": [{"name": "chA"}]})
