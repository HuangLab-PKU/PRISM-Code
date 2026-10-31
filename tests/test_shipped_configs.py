"""Contract tests for the configs shipped in ``code/configs/``.

These pin the interfaces *between* pipeline stages.  The readout stage names
its output intensity columns after the stem of each entry in
``readout.yaml:channel_files``; the gene-calling stage looks those exact names
up as keys of ``channel_correction.channels``.  When the two drift apart the
handoff dies with a KeyError deep inside ``apply_channel_correction``, so the
agreement is asserted here instead.
"""
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load(name):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_readout_channel_names_match_gene_calling_channel_keys():
    from prism.readout.channels import resolve_channels

    readout = _load("readout.yaml")
    gene_calling = _load("gene_calling_base.yaml")

    readout_columns = [name for _, name in resolve_channels(readout)]
    correction_keys = list(gene_calling["channel_correction"]["channels"].keys())

    assert readout_columns == correction_keys, (
        "readout.yaml channel names must match the channel_correction keys in "
        "gene_calling_base.yaml, or the readout -> gene_calling handoff raises "
        "KeyError"
    )


def test_detection_snr_keys_match_channel_names():
    from prism.readout.channels import resolve_channels

    readout = _load("readout.yaml")

    readout_columns = [name for _, name in resolve_channels(readout)]

    assert list(readout["detection_snr"].keys()) == readout_columns


def test_transform_matrix_is_square_and_matches_channel_count():
    gene_calling = _load("gene_calling_base.yaml")
    correction = gene_calling["channel_correction"]

    n = len(correction["channels"])
    matrix = correction["transform_matrix"]

    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)


@pytest.mark.parametrize(
    "name", ["gene_calling.yaml", "gene_calling_full.yaml", "gene_calling_spherical.yaml"]
)
def test_entrypoint_configs_declare_a_classification_method(name):
    """Every config meant to be passed to the pipeline must resolve a method."""
    from prism.gene_calling.config_loader import (
        load_gene_calling_config,
        validate_gene_calling_config,
    )

    config = load_gene_calling_config(CONFIG_DIR / name)

    assert validate_gene_calling_config(config) == []
