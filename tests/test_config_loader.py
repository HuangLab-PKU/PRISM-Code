"""Tests for modular gene-calling configuration loading.

The loader composes a configuration from three layers, in increasing
precedence: ``config_modules`` (in declared order), the main file's own body,
and ``overrides``.  A content file with no ``config_modules`` key must
therefore load as itself -- that is what makes "an experiment is just a
config" possible.
"""
import textwrap

import pytest

from prism.gene_calling.config_loader import load_gene_calling_config


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_standalone_content_file_loads_its_own_body(tmp_path):
    """A config with no config_modules must load as itself, not as {}."""
    cfg_path = _write(
        tmp_path,
        "standalone.yaml",
        """
        classification:
          method: gmm
        prism_panel:
          num_per_layer: 15
        """,
    )

    config = load_gene_calling_config(cfg_path)

    assert config["classification"]["method"] == "gmm"
    assert config["prism_panel"]["num_per_layer"] == 15


def test_modules_merge_in_declared_order(tmp_path):
    _write(tmp_path, "base.yaml", "classification: {method: gmm, tol: 0.001}\n")
    _write(tmp_path, "gmm.yaml", "classification: {tol: 0.05}\n")
    cfg_path = _write(
        tmp_path,
        "main.yaml",
        """
        config_modules:
          - base.yaml
          - gmm.yaml
        """,
    )

    config = load_gene_calling_config(cfg_path)

    # Later module wins on conflict, earlier module's other keys survive.
    assert config["classification"]["method"] == "gmm"
    assert config["classification"]["tol"] == 0.05


def test_main_body_layers_over_modules(tmp_path):
    _write(tmp_path, "base.yaml", "classification: {method: gmm, tol: 0.001}\n")
    cfg_path = _write(
        tmp_path,
        "main.yaml",
        """
        config_modules:
          - base.yaml
        classification:
          method: codebook_gmm
        """,
    )

    config = load_gene_calling_config(cfg_path)

    assert config["classification"]["method"] == "codebook_gmm"
    # Merge must be deep: keys the main body did not mention survive.
    assert config["classification"]["tol"] == 0.001


def test_overrides_beat_main_body_and_modules(tmp_path):
    _write(tmp_path, "base.yaml", "classification: {method: gmm}\n")
    cfg_path = _write(
        tmp_path,
        "main.yaml",
        """
        config_modules:
          - base.yaml
        classification:
          method: codebook_gmm
        overrides:
          classification:
            method: postcode
        """,
    )

    config = load_gene_calling_config(cfg_path)

    assert config["classification"]["method"] == "postcode"


def test_config_modules_and_overrides_are_not_leaked_into_result(tmp_path):
    _write(tmp_path, "base.yaml", "classification: {method: gmm}\n")
    cfg_path = _write(
        tmp_path,
        "main.yaml",
        """
        config_modules:
          - base.yaml
        overrides:
          classification:
            tol: 0.05
        """,
    )

    config = load_gene_calling_config(cfg_path)

    assert "config_modules" not in config
    assert "overrides" not in config


def test_missing_module_fails_fast(tmp_path):
    """A typo'd module name must not silently yield a partial config."""
    cfg_path = _write(
        tmp_path,
        "main.yaml",
        """
        config_modules:
          - does_not_exist.yaml
        """,
    )

    with pytest.raises(FileNotFoundError, match="does_not_exist.yaml"):
        load_gene_calling_config(cfg_path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gene_calling_config(tmp_path / "nope.yaml")
