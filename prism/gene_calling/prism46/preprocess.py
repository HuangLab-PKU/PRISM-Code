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

def run(context):
    """Apply the versioned correction, threshold and three-layer recipe."""
    RUN_ID = context.run_id
    read_dir = context.read_dir
    figure_dir = read_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'text.usetex': False, 'figure.dpi': 300})
    PRISM_PANEL = 'PRISM46'
    Q_CHNS = ['Ye/A', 'B/A', 'R/A']
    Q_NUM = int(COLOR_GRADE * (COLOR_GRADE + 1) / 2)
    XRANGE, YRANGE = ([-0.8, 0.8], [-0.6, 0.8])
    s = 0.05
    alpha = 0.05
    percentile_thre = 99.5
    bins = (500, 500)
    GENOTYPE = context.panel_id
    PANEL = context.panel
    CLASSES = context.classes
    LABEL_TO_PROBE = build_label_to_probe()
    GENES = {lbl: PANEL[probe] for lbl, probe in LABEL_TO_PROBE.items()}
    print(f'  genotype: {GENOTYPE}  (panel column gene_{GENOTYPE})', flush=True)
    print(f'  proteins: ' + ', '.join((f'{code_tag(p)}={PANEL[p]}' for p in sorted(PANEL) if CLASSES[p] == 'protein')), flush=True)
    print(f'  unprinted codewords: ' + ', '.join((code_tag(p) for p in sorted(PANEL) if CLASSES[p] == 'unprinted')), flush=True)
    intensity = pd.read_csv(read_dir / 'intensity.csv', index_col=0)
    _RENAME = {'r01_ex628': 'R', 'cyc_1_cy5': 'R', 'r01_ex586': 'Ye', 'cyc_1_TxRed': 'Ye', 'r01_ex531': 'G', 'cyc_1_cy3': 'G', 'r01_ex485': 'B', 'cyc_1_FAM': 'B'}
    intensity.rename(columns=_RENAME, inplace=True)
    _missing = [c for c in ('R', 'Ye', 'G', 'B') if c not in intensity.columns]
    if _missing:
        raise KeyError(f'intensity.csv is missing channels {_missing} after rename; columns present: {list(intensity.columns)}')
    CT = 0.25
    FG, FB = (0.6, 0.05)
    ARCSINH_COFACTOR = 0.3
    SR_C = SYE_C = SB_C = 1.0
    COLOR_RELIABILITY_FLOOR = 500
    PURE_G_QUANTILE = 0.99
    PURE_G_MARGIN = 1.5
    PURE_G_RATIO = None
    SG_ANCHOR_G_OVER_RAWSUM = [0.16, 0.43]
    SG_ANCHOR_SG = [3.5, 2.0]
    SG_MIN, SG_MAX = (1.5, 4.5)
    SG = None
    SG = context.options.get('sg')
    PURE_G_RATIO = context.options.get('pure_g_ratio')
    _med = {c: float(intensity.loc[intensity[c] > 0, c].median()) for c in ('R', 'Ye', 'G', 'B')}
    _rawsum = intensity['R'] + intensity['Ye'] + intensity['B']
    _g_over_rawsum = float((intensity['G'] / _rawsum.replace(0, np.nan)).median())
    print(f'  detected spots: {len(intensity):,}', flush=True)
    print(f"  channel medians: R={_med['R']:.0f} Ye={_med['Ye']:.0f} B={_med['B']:.0f} G={_med['G']:.0f}", flush=True)
    if SG is None:
        SG = float(np.clip(np.interp(_g_over_rawsum, SG_ANCHOR_G_OVER_RAWSUM, SG_ANCHOR_SG), SG_MIN, SG_MAX))
        _sg_origin = f'interpolated from exp15 anchors {SG_ANCHOR_G_OVER_RAWSUM} -> {SG_ANCHOR_SG}'
    else:
        _sg_origin = 'explicit run configuration'
    print(f'  G/rawsum median = {_g_over_rawsum:.3f}  ->  SG = {SG:.2f} ({_sg_origin})', flush=True)
    _B_raw = intensity['B'].copy()
    intensity['B'] = np.maximum(_B_raw - intensity['G'] * CT, 0)
    intensity['B_c'] = np.maximum(_B_raw - intensity['G'] * CT, 0)
    intensity['Scaled_R'] = intensity['R'] * 1.0
    intensity['Scaled_Ye'] = intensity['Ye'] * 1.0
    intensity['Scaled_G'] = intensity['G'] * SG
    intensity['Scaled_B'] = intensity['B'] * 1.0
    intensity['sum'] = intensity['Scaled_R'] + intensity['Scaled_Ye'] + intensity['Scaled_B']
    intensity['SR_c'] = intensity['R'] * SR_C
    intensity['SYe_c'] = intensity['Ye'] * SYE_C
    intensity['SB_c'] = intensity['B_c'] * SB_C
    intensity['csum'] = intensity['SR_c'] + intensity['SYe_c'] + intensity['SB_c']

    def valley_threshold(values, floor, label):
        """Noise/signal cut: the minimum of a 2-component GMM density on log10(values).

        Args:
            values: Positive intensity-like series.
            floor: Hard lower bound the data-driven cut may not go below.
            label: Name used in the progress line.

        Returns:
            ``(threshold, valley)`` - the applied cut and the raw valley position.
        """
        v = np.asarray(values, dtype=float)
        v = v[v > 0]
        lv = np.log10(v)
        rng = np.random.default_rng(0)
        fit = lv if len(lv) <= 200000 else rng.choice(lv, 200000, replace=False)
        gm = _GMM(2, n_init=4, random_state=0).fit(fit.reshape(-1, 1))
        mu = np.sort(gm.means_.ravel())
        xs = np.linspace(mu[0], mu[1], 1000)
        valley = float(10 ** xs[np.argmin(np.exp(gm.score_samples(xs.reshape(-1, 1))))])
        thr = max(floor, valley)
        print(f'  {label} (data-driven): valley={valley:.0f}, floor={floor:.0f} -> {thr:.0f}', flush=True)
        return (thr, valley)
    SUM_THRESHOLD, _sum_valley = valley_threshold(intensity['sum'], COLOR_RELIABILITY_FLOOR, 'sum threshold')
    G_THRESHOLD, _g_valley = valley_threshold(intensity['G'], COLOR_RELIABILITY_FLOOR / 2.0, f'Cy3 threshold (probe {PURE_G_PROBE})')
    _g_over_sum = intensity['G'] / intensity['sum'].replace(0, np.nan)
    _colour_ok = intensity['sum'] >= SUM_THRESHOLD
    _ratio_ceiling = float(_g_over_sum[_colour_ok].quantile(PURE_G_QUANTILE))
    if PURE_G_RATIO is None:
        PURE_G_RATIO = _ratio_ceiling * PURE_G_MARGIN
        print(f'  pure-Cy3 ratio (derived): q{PURE_G_QUANTILE:g} of G/(R+Ye+B) among {int(_colour_ok.sum()):,} colour-bearing spots = {_ratio_ceiling:.2f}, x{PURE_G_MARGIN} -> PURE_G_RATIO={PURE_G_RATIO:.2f}', flush=True)
    else:
        print(f'  pure-Cy3 ratio (given): {PURE_G_RATIO:.2f} (colour-codeword ceiling is {_ratio_ceiling:.2f})', flush=True)
    _pure_g = ((_g_over_sum > PURE_G_RATIO) | ~np.isfinite(_g_over_sum)) & (intensity['G'] >= G_THRESHOLD)
    intensity_G = intensity[_pure_g].copy()
    intensity = intensity[~_pure_g].copy()
    _pure_bg_frac = float((intensity_G['sum'] < SUM_THRESHOLD).mean()) if len(intensity_G) else float('nan')
    print(f'  pure-Cy3 (probe {PURE_G_PROBE} = {PANEL[PURE_G_PROBE]}, G/(R+Ye+B) > {PURE_G_RATIO} & G >= {G_THRESHOLD:.0f}): {len(intensity_G):,} spots; {100 * _pure_bg_frac:.1f}% of them also sit below SUM_THRESHOLD (colour budget really is background)', flush=True)
    fig, axp = plt.subplots(ncols=2, figsize=(13, 4))
    _r = _g_over_sum.replace([np.inf, -np.inf], np.nan).dropna()
    _r = _r[_r > 0]
    sns.histplot(np.log10(_r), bins=250, ax=axp[0])
    axp[0].axvline(np.log10(PURE_G_RATIO), color='red', ls='--', lw=2, label=f'PURE_G_RATIO={PURE_G_RATIO:.2f}')
    axp[0].axvline(np.log10(_ratio_ceiling), color='orange', ls=':', lw=1.5, label=f'colour-codeword ceiling (q{PURE_G_QUANTILE:g})={_ratio_ceiling:.2f}')
    axp[0].set_yscale('log')
    axp[0].set_xlabel('log10( G / (R+Ye+B) )')
    axp[0].set_title(f'pure-Cy3 selection - {len(intensity_G):,} of {len(_r):,} spots right of the cut')
    axp[0].legend(fontsize=8)
    sns.histplot(intensity.loc[intensity['G'] > 0, 'G'], bins=np.logspace(1, np.log10(60000), 200), ax=axp[1], color='0.6', label='colour path')
    if len(intensity_G):
        sns.histplot(intensity_G.loc[intensity_G['G'] > 0, 'G'], bins=np.logspace(1, np.log10(60000), 200), ax=axp[1], color='green', label=f'probe {PURE_G_PROBE}')
    axp[1].axvline(G_THRESHOLD, color='red', ls='--', lw=1.5, label=f'G_THRESHOLD={G_THRESHOLD:.0f}')
    axp[1].set_xscale('log')
    axp[1].set_yscale('log')
    axp[1].set_title('Cy3 intensity')
    axp[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / 'hist_pure_g.png', dpi=200, bbox_inches='tight')
    plt.close()
    _n_all = len(intensity)
    _n_keep = int((intensity['sum'] >= SUM_THRESHOLD).sum())
    _keep_pct = 100.0 * _n_keep / _n_all
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0, :])
    _pos = intensity['sum'] > 0
    sns.histplot(intensity.loc[_pos, 'sum'], bins=np.logspace(1, np.log10(20000), 250), ax=ax0)
    ax0.set_xscale('log')
    ax0.set_xlim((30, 20000))
    ax0.axvline(SUM_THRESHOLD, color='red', ls='--', lw=2, label=f'SUM_THRESHOLD={SUM_THRESHOLD:.0f}  (kept {_n_keep:,} = {_keep_pct:.1f}%)')
    ax0.axvline(_sum_valley, color='green', ls=':', lw=1.5, label=f'noise/signal valley={_sum_valley:.0f}')
    ax0.axvline(COLOR_RELIABILITY_FLOOR, color='orange', ls=':', lw=1.5, label=f'colour floor={COLOR_RELIABILITY_FLOOR}')
    ax0.legend(fontsize=9)
    ax0.set_title(f'Intensity Sum - {_n_all:,} colour-path points (sum = R+Ye+B, G excluded)')
    for i, channel in enumerate(['R', 'Ye', 'B', 'G']):
        ax = plt.subplot(gs[1, i])
        sns.histplot(intensity.loc[intensity[channel] > 0, channel], bins=300, ax=ax)
        median = intensity.loc[intensity[channel] > 0, channel].median()
        ax.set_xlim((0, 10000))
        ax.set_title(f'{channel}_{median:.0f} (all pts)')
    plt.tight_layout()
    plt.savefig(figure_dir / 'hist_intensity.png')
    plt.close()
    _initial_n = len(intensity)
    intensity = intensity[intensity['sum'] >= SUM_THRESHOLD].copy()
    print(f'  intensity sum filter (>= {SUM_THRESHOLD:.0f}): {_initial_n:,} -> {len(intensity):,} ({100 * len(intensity) / _initial_n:.1f}% kept)', flush=True)
    intensity['Ye/A'] = intensity['SYe_c'] / intensity['csum']
    intensity['B/A'] = intensity['SB_c'] / intensity['csum']
    intensity['R/A'] = intensity['SR_c'] / intensity['csum']
    intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']
    intensity = intensity[np.isfinite(intensity['G/A'])].copy()
    intensity['G/A'] = intensity['G/A'] * np.exp(FG * intensity['Scaled_Ye'] / intensity['sum'])
    intensity['B/A'] = intensity['B/A'] * np.exp(FB * intensity['G/A'])
    intensity['G/A'] = np.arcsinh(intensity['G/A'] / ARCSINH_COFACTOR)
    intensity['sum_scale'] = intensity['R/A'] + intensity['Ye/A'] + intensity['B/A']
    intensity['R/A'] = intensity['R/A'] / intensity['sum_scale']
    intensity['Ye/A'] = intensity['Ye/A'] / intensity['sum_scale']
    intensity['B/A'] = intensity['B/A'] / intensity['sum_scale']
    CD_1_proj = np.array([[-np.sqrt(2) / 2], [np.sqrt(2) / 2], [0]])
    CD_2_proj = np.array([[-1 / 2], [-1 / 2], [np.sqrt(2) / 2]])
    RYB_xy_transform = np.concatenate([CD_1_proj, CD_2_proj], axis=1)
    intensity['CD_1'] = intensity[Q_CHNS] @ CD_1_proj
    intensity['CD_2'] = intensity[Q_CHNS] @ CD_2_proj
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 5))
    for a, col in zip(ax, ['Ye/A', 'B/A', 'R/A', 'G/A']):
        a.hist(bins=100, x=intensity[col])
        a.set_yscale('log')
    plt.savefig(figure_dir / 'hist_raw.png', dpi=300, bbox_inches='tight')
    plt.close()
    finite_mask = np.isfinite(intensity[['CD_1', 'CD_2', 'G/A']].values).all(axis=1)
    intensity = intensity[finite_mask]

    def plot_hist_with_extrema(a, ax=None, bins=100, extrema='max', kde_kws=None):
        if kde_kws is None:
            kde_kws = {'bw_adjust': 0.5}
        sns.histplot(a, bins=bins, stat='count', edgecolor='white', alpha=1, ax=ax, kde=True, kde_kws=kde_kws)
        lines = ax.get_lines()
        if not lines:
            return []
        y = lines[0].get_ydata()
        if extrema == 'max':
            y = -y
        extrema = [float(_ / len(y) * (max(a) - min(a)) + min(a)) for _ in argrelextrema(np.array(y), np.less)[0]]
        for subextrema in extrema:
            ax.axvline(x=subextrema, color='r', alpha=0.5, linestyle='--')
        return extrema
    THEORY_GRID = [i / (COLOR_GRADE - 1) for i in range(COLOR_GRADE)]
    _plot_sample = intensity.sample(min(len(intensity), 200000), random_state=0)
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 7))
    plt.setp(ax, xlim=(-0.2, 1.2))
    Y_maxima = plot_hist_with_extrema(_plot_sample['Ye/A'], ax=ax[0], extrema='max', kde_kws={'bw_adjust': 1.6})
    B_maxima = plot_hist_with_extrema(_plot_sample['B/A'], ax=ax[1], extrema='max', kde_kws={'bw_adjust': 1})
    R_maxima = plot_hist_with_extrema(_plot_sample['R/A'], ax=ax[2], extrema='max', kde_kws={'bw_adjust': 1})
    plot_hist_with_extrema(_plot_sample['G/A'], ax=ax[3], extrema='min', kde_kws={'bw_adjust': 0.8})
    print(f'  empirical colour maxima: Ye={[round(v, 3) for v in Y_maxima]} B={[round(v, 3) for v in B_maxima]} R={[round(v, 3) for v in R_maxima]}', flush=True)
    _fallback = []
    if len(Y_maxima) != COLOR_GRADE:
        Y_maxima = THEORY_GRID
        _fallback.append('Ye')
    if len(B_maxima) != COLOR_GRADE:
        B_maxima = THEORY_GRID
        _fallback.append('B')
    if len(R_maxima) != COLOR_GRADE:
        R_maxima = THEORY_GRID
        _fallback.append('R')
    if _fallback:
        print(f'  KDE did not give {COLOR_GRADE} maxima for {_fallback}; using theoretical grid {THEORY_GRID} for those channels', flush=True)
    for a in ax:
        a.set_yscale('log')
    plt.savefig(figure_dir / 'hist_chn.png', dpi=300, bbox_inches='tight')
    plt.close()
    _ga_arr = intensity['G/A'].values
    minima, _g_diag = choose_g_layer_cuts(_ga_arr, GLAYER)
    _g_q05, _g_q95 = (_g_diag['q_lo'], _g_diag['q_hi'])
    G_minima = _g_diag['valleys']
    print(f"  G/A modes: {[round(v, 3) for v in _g_diag['peaks']]} (all KDE valleys: {[round(v, 3) for v in G_minima]}; min mode separation {_g_diag['min_separation']:.3f})", flush=True)
    if len(minima) < GLAYER - 1:
        _g2 = _GMM(GLAYER, n_init=4, random_state=0).fit(_ga_arr.reshape(-1, 1))
        _grid = np.linspace(_ga_arr.min(), _ga_arr.max(), 4000).reshape(-1, 1)
        _pred = _g2.predict(_grid)
        _bounds = [float(_grid[i, 0]) for i in range(1, len(_grid)) if _pred[i] != _pred[i - 1]]
        minima = _bounds[:GLAYER - 1]
        print(f"  G_layer cuts from GMM fallback (only {len(_g_diag['peaks'])} modes found, need {GLAYER}): {minima}", flush=True)
    if len(minima) < GLAYER - 1:
        raise RuntimeError(f'Could not find {GLAYER - 1} G-layer cut point(s); G/A q5={_g_q05:.3f} q95={_g_q95:.3f}. Inspect figures/hist_chn.png and set SG explicitly.')
    print(f'  G_layer cut points: {[round(m, 3) for m in minima]} (G/A q5={_g_q05:.3f} q95={_g_q95:.3f})', flush=True)
    minima = list(minima)
    minima.insert(0, intensity['G/A'].min() - 0.01)
    minima.append(intensity['G/A'].max() + 0.01)
    intensity['G_layer'] = pd.cut(intensity['G/A'], bins=minima, labels=[_ for _ in range(len(minima) - 1)], include_lowest=True, right=False)
    _layer_counts = intensity['G_layer'].value_counts().sort_index()
    print(f'  G_layer occupancy: {dict(_layer_counts)}', flush=True)

    def ybrg_to_rgb(ybr, g=0):
        y, b, r = ybr
        red = y + r
        green = 0.9 * y + 0.2 * g
        blue = b
        return ((red, green, blue) / np.max((red, green, blue))).clip(0, 1)
    GRADES_BY_SLOT = grades_by_slot(COLOR_GRADE)
    centroid_init_dict = dict()
    colormap = dict()
    fig, ax = plt.subplots(nrows=3, ncols=GLAYER, figsize=(20, 10))
    for layer in range(GLAYER):
        data = intensity[intensity['G_layer'] == layer]
        data = data.sample(min(100000, len(data)), random_state=0)
        ax_tmp = ax if GLAYER < 2 else ax[:, layer]
        ax_tmp[0].set_title(f'G_layer{layer}')
        Y_max_l, B_max_l, R_max_l = (Y_maxima, B_maxima, R_maxima)
        if len(data):
            _y = plot_hist_with_extrema(data['Ye/A'], ax=ax_tmp[0], extrema='max', kde_kws={'bw_adjust': 0.8})
            _b = plot_hist_with_extrema(data['B/A'], ax=ax_tmp[1], extrema='max', kde_kws={'bw_adjust': 0.9})
            _r = plot_hist_with_extrema(data['R/A'], ax=ax_tmp[2], extrema='max', kde_kws={'bw_adjust': 0.7})
            Y_max_l = _y if len(_y) == COLOR_GRADE else Y_maxima
            B_max_l = _b if len(_b) == COLOR_GRADE else B_maxima
            R_max_l = _r if len(_r) == COLOR_GRADE else R_maxima
        centroid_init_dict[layer] = np.array([[Y_max_l[_ye], B_max_l[_b], R_max_l[_r]] for _ye, _b, _r in GRADES_BY_SLOT])
        color_list = [ybrg_to_rgb(_, g=layer / GLAYER) for _ in centroid_init_dict[layer]]
        colormap[layer] = {layer * Q_NUM + i + 1: color_list[i] for i in range(len(color_list))}
        for a in ax_tmp:
            a.set_yscale('log')
    plt.savefig(figure_dir / 'hist_chn_layer.png', dpi=300, bbox_inches='tight')
    plt.close()
    G_A_max = float(intensity['G/A'].max()) * 1.02
    n_cols = 2 + GLAYER
    fig, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(5 * n_cols + 2, 10))
    _scatter_sample = intensity.sample(min(len(intensity), 400000), random_state=0)
    ax[1, 0].scatter(_scatter_sample['CD_1'], _scatter_sample['G/A'], s=s, alpha=alpha, linewidths=None)
    ax[1, 0].set_ylim([-0.2, G_A_max])
    ax[0, 1].scatter(_scatter_sample['CD_2'], _scatter_sample['G/A'], s=s, alpha=alpha, linewidths=None)
    ax[0, 1].set_ylim([-0.2, G_A_max])
    ax[0, 0].scatter(_scatter_sample['CD_1'], _scatter_sample['CD_2'], s=s, alpha=alpha, linewidths=None)
    ax[0, 0].set_xlim(XRANGE)
    ax[0, 0].set_ylim(YRANGE)
    for subextrema in minima[1:-1]:
        ax[1, 0].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
        ax[0, 1].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
    for layer in range(GLAYER):
        ax_scatter = ax[0, 2 + layer]
        ax_hist = ax[1, 2 + layer]
        sub = intensity[intensity['G_layer'] == layer]
        ax_scatter.set_title(f'G={layer}  (n={len(sub):,})')
        _sub_s = sub.sample(min(len(sub), 300000), random_state=0)
        ax_scatter.scatter(_sub_s['CD_1'], _sub_s['CD_2'], s=s, alpha=alpha, linewidths=None)
        ax_scatter.set_xlim(XRANGE)
        ax_scatter.set_ylim(YRANGE)
        for i in range(Q_NUM):
            lbl = layer * Q_NUM + i + 1
            probe = LABEL_TO_PROBE[lbl]
            c = centroid_init_dict[layer][i]
            cx, cy = ((c @ CD_1_proj).item(), (c @ CD_2_proj).item())
            _col = 'grey' if CLASSES[probe] == 'unprinted' else 'cyan'
            ax_scatter.plot(cx, cy, 'o', mfc='none', mec=_col, ms=12, mew=1.2)
            ax_scatter.text(cx, cy + 0.05, f'{code_tag(probe)}:{display_name(probe, GENES[lbl])}', color=_col, fontsize=6, ha='center')
        x, y = (sub['CD_1'], sub['CD_2'])
        if len(sub):
            hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
            percentile = np.percentile(hist, percentile_thre)
            ax_hist.hist2d(x, y, bins=bins, vmax=percentile, range=[XRANGE, YRANGE], cmap='inferno')
        ax_hist.set_xlim(XRANGE)
        ax_hist.set_ylim(YRANGE)
    plt.savefig(figure_dir / 'ColorSpace_3view.png', dpi=300, bbox_inches='tight')
    plt.close()
    params = {'PRISM_PANEL': PRISM_PANEL, 'GENOTYPE': GENOTYPE, 'GLAYER': GLAYER, 'COLOR_GRADE': COLOR_GRADE, 'Q_CHNS': Q_CHNS, 'Q_NUM': Q_NUM, 'XRANGE': XRANGE, 'YRANGE': YRANGE, 's': s, 'alpha': alpha, 'percentile_thre': percentile_thre, 'bins': bins, 'CD_1_proj': CD_1_proj.tolist(), 'CD_2_proj': CD_2_proj.tolist(), 'centroid_init_dict': centroid_init_dict, 'colormap': colormap, 'PANEL': PANEL, 'CLASSES': CLASSES, 'LABEL_TO_PROBE': LABEL_TO_PROBE, 'GENES': GENES, 'PURE_G_PROBE': PURE_G_PROBE, 'PURE_G_RATIO': float(PURE_G_RATIO), 'PURE_G_RATIO_CEILING': _ratio_ceiling, 'PURE_G_QUANTILE': PURE_G_QUANTILE, 'PURE_G_MARGIN': PURE_G_MARGIN, 'SG': float(SG), 'SG_origin': _sg_origin, 'SUM_THRESHOLD': float(SUM_THRESHOLD), 'G_THRESHOLD': float(G_THRESHOLD), 'CT': CT, 'FG': FG, 'FB': FB, 'ARCSINH_COFACTOR': ARCSINH_COFACTOR, 'colour_maxima': {'Ye': [float(v) for v in Y_maxima], 'B': [float(v) for v in B_maxima], 'R': [float(v) for v in R_maxima]}, 'G_layer_cuts': [float(m) for m in minima[1:-1]], 'G_kde_valleys': [float(v) for v in G_minima], 'G_modes': [float(v) for v in _g_diag['peaks']], 'channel_medians': _med, 'G_over_rawsum_median': _g_over_rawsum, 'n_detected': int(_n_all + len(intensity_G)), 'n_pure_g': int(len(intensity_G))}
    with open(read_dir / 'params.yaml', 'w') as f:
        params['confidence_threshold'] = context.options.get('confidence_threshold', 0.8)
        yaml.safe_dump(plain(params), f, sort_keys=False)
    intensity.to_csv(read_dir / 'intensity_preprocessed.csv')
    intensity_G.to_csv(read_dir / 'intensity_G.csv')
    print(f'  wrote intensity_preprocessed.csv ({len(intensity):,} spots)', flush=True)
