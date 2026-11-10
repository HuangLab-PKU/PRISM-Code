"""Raster and coordinate contracts for figures used to draw manual gates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from prism.gene_calling.base import ClassificationResult
from prism.gene_calling.evaluator import ClassificationEvaluator
from prism.gene_calling.layer_figures import export_manual_gate_layers


def test_pipeline_exports_screen_sized_pairs_and_exact_mapping(tmp_path):
    data = pd.DataFrame({'color_dim_1': [-.3, .2, .1],
                         'color_dim_2': [.1, .3, -.1],
                         'ch3/A': [.2, 1., 1.], 'G_layer': [0, 1, 1]})
    result = ClassificationResult(labels=np.array([1, 16, -1]))
    # A user-wide tight-save setting must not change the gate's pixel extent.
    with plt.rc_context({'savefig.bbox': 'tight'}):
        ClassificationEvaluator({}).visualize_channel_space(
            result, data, tmp_path, g_layer_num=3)
    for layer in (1, 2, 3):
        for row in (1, 2):
            with Image.open(tmp_path / f'{row}-layer{layer}.png') as image:
                assert image.size == (800, 700)
    meta = yaml.safe_load((tmp_path / 'layer_figures.yaml').read_text())
    assert [meta['layers'][i]['n_spots'] for i in (1, 2, 3)] == [1, 1, 0]
    assert meta['XRANGE'][0] + 800 * meta['px_size'] == pytest.approx(.8)
    assert meta['YRANGE'][1] - 700 * meta['px_size'] == pytest.approx(-.6)
    # First/last pixel centers remain inside the intended coordinates.
    assert -.8 + .5 * meta['px_size'] == pytest.approx(-.799)
    assert .8 - 699.5 * meta['px_size'] == pytest.approx(-.599)


def test_gate_export_keeps_all_assigned_rows(tmp_path):
    n = 200_001
    data = pd.DataFrame({'color_dim_1': np.zeros(n), 'color_dim_2': np.zeros(n)})
    export_manual_gate_layers(data, np.ones(n, dtype=int),
                              [np.ones(n, dtype=bool)], tmp_path,
                              {1: '#0072B2'}, point_size=.05)
    meta = yaml.safe_load((tmp_path / 'layer_figures.yaml').read_text())
    assert meta['sampling'] == 'none'
    assert meta['layers'][1]['n_spots'] == n


@pytest.mark.parametrize('bad', ['length', 'nonfinite'])
def test_rejects_unmappable_input(tmp_path, bad):
    data = pd.DataFrame({'color_dim_1': [np.nan if bad == 'nonfinite' else 0.],
                         'color_dim_2': [0.]})
    labels = np.array([1, 2] if bad == 'length' else [1])
    with pytest.raises(ValueError):
        export_manual_gate_layers(data, labels, [np.array([True])],
                                  tmp_path, {1: '#0072B2'})
