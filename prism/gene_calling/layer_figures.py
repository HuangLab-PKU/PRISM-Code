"""Screen-sized colour-space rasters with exact manual-gate coordinates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

WIDTH_PX = 800
HEIGHT_PX = 700
DPI = 120
XRANGE = (-0.8, 0.8)
YRANGE = (-0.6, 0.8)


def export_manual_gate_layers(data, labels, layer_masks, output_dir,
                              label_to_color, *, point_size=0.2, alpha=0.25,
                              label_names=None, seeds=None, bins=200, percentile=95,
                              run_id="", layer_cuts=()):
    """Write scatter/density pairs from every assigned spot, without sampling.

    ``layer_masks`` must describe the actual classifier layers, or explicitly
    selected label groups. Labels are cluster IDs, never inferred probe identities.
    PNG edges are exactly XRANGE/YRANGE; no titles or tight bounding box are added.
    The sidecar maps top-left-origin raster pixel centers back to colour space.
    """
    labels = np.asarray(labels)
    xy = data[['color_dim_1', 'color_dim_2']].to_numpy()
    masks = [np.asarray(mask, dtype=bool) for mask in layer_masks]
    if labels.shape != (len(data),) or any(m.shape != labels.shape for m in masks):
        raise ValueError('Labels and layer masks must match the data row count')
    if not np.isfinite(xy).all():
        raise ValueError('Manual-gate coordinates must be finite')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written, layers = [], {}
    # Fixed physical font size at 120 dpi stays readable at native screen scale.
    with plt.rc_context({'font.family': 'Arial', 'text.usetex': False,
                         'savefig.bbox': None}):
        for k, mask in enumerate(masks, start=1):
            selected = mask & (labels != -1)
            points, cluster_ids = xy[selected], labels[selected]
            layers[k] = {'n_spots': int(selected.sum()),
                         'cluster_labels': np.unique(cluster_ids).tolist(), 'G_layer': k - 1}
            for row in (1, 2):
                fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
                try:
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.set(xlim=XRANGE, ylim=YRANGE)
                    ax.set_axis_off()
                    if row == 1 and len(points):
                        for label in np.unique(cluster_ids):
                            group = points[cluster_ids == label]
                            ax.scatter(group[:, 0], group[:, 1],
                                       color=label_to_color[label], s=point_size,
                                       alpha=alpha, marker='.')
                            ax.text(*group.mean(axis=0), (label_names or {}).get(label, str(label)), fontsize=8,
                                    ha='center', va='center', clip_on=True)
                        if seeds is not None and k - 1 in seeds:
                            seed = np.asarray(seeds[k - 1])
                            ax.scatter(seed[:, 0], seed[:, 1], color='cyan', s=1.5, alpha=.7)
                    elif row == 2:
                        fig.patch.set_facecolor('black')
                        if len(points):
                            hist, _, _ = np.histogram2d(
                                points[:, 0], points[:, 1], bins=bins,
                                range=[XRANGE, YRANGE])
                            ax.hist2d(points[:, 0], points[:, 1], bins=bins,
                                      range=[XRANGE, YRANGE], cmap='inferno',
                                      vmax=max(float(np.percentile(hist, percentile)), 1.0))
                    path = output_dir / f'{row}-layer{k}.png'
                    fig.savefig(path, dpi=DPI, bbox_inches=None, pad_inches=0)
                    written.append(path)
                finally:
                    plt.close(fig)
    sidecar = {
        'run_id': run_id, 'G_layer_cuts': list(layer_cuts),
        'width_px': WIDTH_PX, 'height_px': HEIGHT_PX, 'dpi': DPI,
        'XRANGE': list(XRANGE), 'YRANGE': list(YRANGE),
        'px_size': (XRANGE[1] - XRANGE[0]) / WIDTH_PX,
        'pixel_to_colourspace':
            'CD_1 = XRANGE[0] + (col + 0.5) * px_size; '
            'CD_2 = YRANGE[1] - (row + 0.5) * px_size',
        'sampling': 'none', 'unassigned_label': -1,
        'label_identity': 'classifier cluster ID, not inferred probe number',
        'layers': layers,
    }
    (output_dir / 'layer_figures.yaml').write_text(
        yaml.safe_dump(sidecar, sort_keys=False), encoding='utf-8')
    return written
