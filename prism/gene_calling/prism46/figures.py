"""Use the shared manual-gating exporter with explicit PRISM46 annotations."""
import numpy as np
from ..layer_figures import export_manual_gate_layers
from .codebook import code_tag, display_name


def save_layer_figures(intensity, params, figure_dir, run_id=''):
    labels = intensity['label'].to_numpy()
    data = intensity.rename(columns={'CD_1': 'color_dim_1', 'CD_2': 'color_dim_2'})
    masks = [(intensity['G_layer'].to_numpy() == layer) for layer in range(params['GLAYER'])]
    mapping = {int(k): int(v) for k, v in params['LABEL_TO_PROBE'].items()}
    names = {label: f'{code_tag(probe)}\n{display_name(probe, params["GENES"][label])}'
             for label, probe in mapping.items()}
    colors = {int(label): rgb for layer in params['colormap'].values()
              for label, rgb in layer.items()}
    projection = np.concatenate([params['CD_1_proj'], params['CD_2_proj']], axis=1)
    seeds = {int(layer): np.asarray(value) @ projection
             for layer, value in params['centroid_init_dict'].items()}
    return export_manual_gate_layers(data, labels, masks, figure_dir, colors,
                                    point_size=params['s'], alpha=params['alpha'],
                                    label_names=names, seeds=seeds,
                                    bins=tuple(params['bins']), percentile=98,
                                    run_id=run_id, layer_cuts=params['G_layer_cuts'])
