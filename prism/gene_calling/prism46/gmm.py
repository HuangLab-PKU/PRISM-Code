from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture as _GMM
from .colorspace import choose_g_layer_cuts
from .codebook import COLOR_GRADE, GLAYER, PURE_G_PROBE, PROBE_GRADES, UNPRINTED, build_label_to_probe, code_tag, display_name, grades_by_slot
from .context import plain
from sklearn.mixture import GaussianMixture as GMM

def apply_gmm(reduced_features, num_clusters, means_init=None):
    gmm = GMM(n_components=num_clusters, covariance_type='diag', means_init=means_init, random_state=0)
    gmm_clusters = gmm.fit(reduced_features)
    labels = gmm_clusters.predict(reduced_features)
    confidence = gmm_clusters.predict_proba(reduced_features).max(axis=1)
    return (gmm_clusters, labels, confidence)

def run(context):
    """Fit seeded colour-grid mixtures and export all layer figures."""
    RUN_ID = context.run_id
    read_dir = context.read_dir
    figure_dir = read_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'text.usetex': False, 'figure.dpi': 300})
    with open(read_dir / 'params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    GLAYER = params['GLAYER']
    Q_CHNS = params['Q_CHNS']
    Q_NUM = params['Q_NUM']
    XRANGE = params['XRANGE']
    YRANGE = params['YRANGE']
    s = params['s']
    alpha = params['alpha']
    CD_1_proj = np.array(params['CD_1_proj'])
    CD_2_proj = np.array(params['CD_2_proj'])
    centroid_init_dict = {int(k): np.asarray(v) for k, v in params['centroid_init_dict'].items()}
    colormap = params['colormap']
    LABEL_TO_PROBE = {int(k): int(v) for k, v in params['LABEL_TO_PROBE'].items()}
    GENES = {int(k): v for k, v in params['GENES'].items()}
    from .codebook import code_tag as _code_tag, display_name as _display_name

    def code_tag(label):
        """GMM label -> 'PR26'. Annotations always carry the PROBE number."""
        return _code_tag(LABEL_TO_PROBE[label]) if label in LABEL_TO_PROBE else str(label)

    def gene_label(label):
        """Gene name for a GMM label, or an explicit marker for an unprinted codeword."""
        if label not in LABEL_TO_PROBE:
            return ''
        return _display_name(LABEL_TO_PROBE[label], GENES.get(label, ''))
    CONFIDENCE_THRESHOLD = float(params.get('confidence_threshold', 0.8))
    intensity = pd.read_csv(read_dir / 'intensity_preprocessed.csv')
    intensity['label'] = -1
    intensity['confidence'] = np.nan
    GMM_dict = dict()
    for layer in range(GLAYER):
        centroids_init = centroid_init_dict[layer]
        filtered_data = intensity[intensity['G_layer'] == layer]
        if len(filtered_data) < len(centroids_init):
            print(f'  layer={layer}: only {len(filtered_data)} samples (< {len(centroids_init)} components); skipping GMM, labels stay -1', flush=True)
            continue
        reduced_features = filtered_data[Q_CHNS]
        gmm, gmm_labels, gmm_conf = apply_gmm(reduced_features, num_clusters=len(centroids_init), means_init=centroids_init)
        GMM_dict[layer] = gmm
        intensity.loc[filtered_data.index, 'label'] = gmm_labels + int(layer * Q_NUM + 1)
        intensity.loc[filtered_data.index, 'confidence'] = gmm_conf
        print(f'  layer={layer}: {len(filtered_data):,} spots -> {len(centroids_init)} components', flush=True)
    bins = (500, 500)
    percentile_thre = 98
    CD_COL_1, CD_COL_2 = ('CD_1', 'CD_2')
    PLOT_MAX = 400000
    fig, ax = plt.subplots(nrows=2, ncols=GLAYER, figsize=(5.5 * GLAYER, 10))
    for layer in range(GLAYER):
        ax_gmm = ax[0] if GLAYER < 2 else ax[0, layer]
        ax_hist = ax[1] if GLAYER < 2 else ax[1, layer]
        data = intensity[intensity['G_layer'] == layer]
        data = data[data['label'] != -1]
        if layer not in GMM_dict or len(data) == 0:
            for a in (ax_gmm, ax_hist):
                a.set_title(f'G={layer} (empty)')
                a.set_xlim(XRANGE)
                a.set_ylim(YRANGE)
            continue
        _plot = data.sample(min(len(data), PLOT_MAX), random_state=0)
        ax_gmm.scatter(_plot[CD_COL_1], _plot[CD_COL_2], c=[colormap[layer][label] for label in _plot['label']], marker='.', alpha=alpha, s=s)
        for i in range(1 + layer * Q_NUM, 1 + (layer + 1) * Q_NUM):
            cen_tmp = np.mean(data[data['label'] == i][[CD_COL_1, CD_COL_2]], axis=0)
            if not np.isfinite(cen_tmp).all():
                continue
            ax_gmm.text(cen_tmp.iloc[0], cen_tmp.iloc[1], f'{code_tag(i)}\n{gene_label(i)}', fontsize=8, color='black', ha='center', va='center')
        RYB_xy_transform = np.concatenate([CD_1_proj, CD_2_proj], axis=1)
        centroid_init = centroid_init_dict[layer] @ RYB_xy_transform
        ax_gmm.scatter(centroid_init[:, 0], centroid_init[:, 1], color='cyan', s=1.5, alpha=0.7)
        ax_gmm.set_title(f'G={layer}')
        ax_gmm.set_xlim(XRANGE)
        ax_gmm.set_ylim(YRANGE)
        x, y = (data[CD_COL_1], data[CD_COL_2])
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        percentile = np.percentile(hist, percentile_thre)
        ax_hist.hist2d(x, y, bins=bins, vmax=percentile, range=[XRANGE, YRANGE], cmap='inferno')
        ax_hist.set_xlim(XRANGE)
        ax_hist.set_ylim(YRANGE)
    plt.tight_layout()
    plt.savefig(figure_dir / 'ColorSpace_GMM.png', dpi=300)
    plt.close()
    from .figures import save_layer_figures
    _layer_pngs = save_layer_figures(intensity, params, figure_dir, run_id=RUN_ID)
    print(f'  wrote {len(_layer_pngs)} per-layer figures', flush=True)
    _labeled = intensity['label'] != -1
    _conf_ok = intensity['confidence'] >= CONFIDENCE_THRESHOLD
    _n_lab = int(_labeled.sum())
    _n_conf = int((_labeled & _conf_ok).sum())
    _mean_conf = float(intensity.loc[_labeled, 'confidence'].mean()) if _n_lab else float('nan')
    print(f'  posterior confidence: mean={_mean_conf:.3f}; kept @ >= {CONFIDENCE_THRESHOLD}: {_n_conf:,}/{_n_lab:,} ({100 * _n_conf / max(_n_lab, 1):.1f}%)', flush=True)
    fig, ax = plt.subplots(nrows=1, ncols=GLAYER, figsize=(5.5 * GLAYER, 5))
    for layer in range(GLAYER):
        a = ax[layer] if GLAYER > 1 else ax
        data = intensity[(intensity['G_layer'] == layer) & _labeled & _conf_ok]
        if layer in GMM_dict and len(data) > 0:
            _plot = data.sample(min(len(data), PLOT_MAX), random_state=0)
            a.scatter(_plot[CD_COL_1], _plot[CD_COL_2], marker='.', alpha=alpha, s=s, c=[colormap[layer][label] for label in _plot['label']])
        a.set_title(f'G={layer}  conf>={CONFIDENCE_THRESHOLD}  (n={len(data):,})')
        a.set_xlim(XRANGE)
        a.set_ylim(YRANGE)
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig(figure_dir / 'ColorSpace_GMM_confident.png', dpi=300)
    plt.close()
    intensity.to_csv(read_dir / 'intensity_labeled.csv', index=False)
    print(f'  wrote intensity_labeled.csv ({len(intensity):,} spots)', flush=True)
