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
from ..prism46.colorspace import choose_g_layer_cuts
from .codebook import COLOR_GRADE, GLAYER, PURE_G_PROBE, PROBE_GRADES, UNPRINTED, build_label_to_probe, code_tag, display_name, grades_by_slot
from ..prism46.context import plain

def run(context):
    """Annotate calls using the explicit panel and keep unprinted controls."""
    RUN_ID = context.run_id
    read_dir = context.read_dir
    work_dir = context.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = read_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'text.usetex': False, 'figure.dpi': 300})
    BIN_PX = int(context.options.get('bin_px', 100))
    with open(work_dir / 'params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    GENOTYPE = params['GENOTYPE']
    GLAYER = params['GLAYER']
    Q_NUM = params['Q_NUM']
    PANEL = {int(k): v for k, v in params['PANEL'].items()}
    CLASSES = {int(k): v for k, v in params['CLASSES'].items()}
    LABEL_TO_PROBE = {int(k): int(v) for k, v in params['LABEL_TO_PROBE'].items()}
    GENES = {int(k): v for k, v in params['GENES'].items()}
    PURE_G_PROBE = int(params['PURE_G_PROBE'])
    CONFIDENCE_THRESHOLD = float(params.get('confidence_threshold', 0.8))
    N_LABELS = GLAYER * Q_NUM
    position = pd.read_csv(read_dir / 'position.csv')
    labeled = pd.read_csv(work_dir / 'intensity_labeled.csv')
    labeled = labeled.merge(position, on='index', how='left')
    labeled['probe'] = labeled['label'].map(LABEL_TO_PROBE)
    labeled['gene'] = labeled['label'].map(GENES)
    pure_g = pd.read_csv(work_dir / 'intensity_G.csv')
    if 'index' not in pure_g.columns:
        pure_g = pure_g.rename(columns={pure_g.columns[0]: 'index'})
    pure_g = pure_g.merge(position, on='index', how='left')
    pure_g['label'] = -1
    pure_g['confidence'] = np.nan
    pure_g['probe'] = PURE_G_PROBE
    pure_g['gene'] = PANEL[PURE_G_PROBE]
    pure_g['G_layer'] = np.nan
    called = pd.concat([labeled[labeled['label'] != -1], pure_g], ignore_index=True, sort=False)
    n_pre = len(labeled)
    n_called = int((labeled['label'] != -1).sum())
    n_conf = int((labeled['confidence'] >= CONFIDENCE_THRESHOLD).sum())
    print(f'  genotype                   : {GENOTYPE}')
    print(f'  spots on the colour path   : {n_pre:,}')
    print(f'  assigned a colour codeword : {n_called:,} ({100 * n_called / max(n_pre, 1):.1f}%)')
    print(f'  confident (>= {CONFIDENCE_THRESHOLD})       : {n_conf:,} ({100 * n_conf / max(n_called, 1):.1f}% of called)')
    print(f'  probe {PURE_G_PROBE} ({PANEL[PURE_G_PROBE]}, pure Cy3) : {len(pure_g):,}')
    print(f'  total gene calls           : {len(called):,}')
    per_probe = {}
    for probe in sorted(PANEL):
        gene = PANEL[probe]
        name = display_name(probe, gene)
        sub = called[called['probe'] == probe]
        conf = sub['confidence']
        per_probe[code_tag(probe)] = {'probe': int(probe), 'name': name, 'gene': gene, 'class': CLASSES[probe], 'G_layer': int((probe - 1) // Q_NUM) if probe <= N_LABELS else None, 'n_spots': int(len(sub)), 'pct_of_calls': round(100.0 * len(sub) / max(len(called), 1), 3), 'pct_of_grid_calls': None, 'n_confident': int((conf >= CONFIDENCE_THRESHOLD).sum()), 'median_confidence': round(float(conf.median()), 4) if conf.notna().any() else None, 'median_sum': round(float(sub['sum'].median()), 1) if 'sum' in sub.columns and len(sub) else None}
    _n_grid_calls = max(int((called['probe'] <= N_LABELS).sum()), 1)
    for _code, _rec in per_probe.items():
        _rec['pct_of_grid_calls'] = round(100.0 * _rec['n_spots'] / _n_grid_calls, 3) if _rec['probe'] <= N_LABELS else None
    summary_df = pd.DataFrame(per_probe).T
    summary_df.index.name = 'code'
    context.log_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(context.log_dir / 'gene_counts.csv')
    print('\n  ' + summary_df[['probe', 'name', 'gene', 'class', 'n_spots', 'pct_of_calls', 'n_confident', 'median_confidence']].to_string().replace('\n', '\n  '))
    _grid_calls = called[called['probe'] <= N_LABELS]
    _unprinted = [p for p in PANEL if p <= N_LABELS and CLASSES[p] == 'unprinted']
    _printed = [p for p in PANEL if p <= N_LABELS and CLASSES[p] != 'unprinted']
    _n_grid = max(len(_grid_calls), 1)
    _unprinted_n = int(_grid_calls['probe'].isin(_unprinted).sum())
    _unprinted_share = 100.0 * _unprinted_n / _n_grid
    _per_unprinted = _unprinted_share / max(len(_unprinted), 1)
    _expected = 100.0 / N_LABELS
    print(f"\n  UNPRINTED CODEWORD CONTROL ({len(_unprinted)} slots: {', '.join((code_tag(p) for p in _unprinted))})")
    print(f'    calls on unprinted codewords : {_unprinted_n:,} of {_n_grid:,} colour-grid calls = {_unprinted_share:.2f}%')
    print(f'    per unprinted codeword       : {_per_unprinted:.3f}%  (uniform would be {_expected:.2f}%)')
    _below = [p for p in _printed if per_probe[code_tag(p)]['pct_of_grid_calls'] <= _per_unprinted]
    _leak = []
    for probe in _unprinted:
        cy5, txred, cy3, fam = PROBE_GRADES[probe]
        _leak.append((fam, probe, txred, fam, cy5, per_probe[code_tag(probe)]['pct_of_grid_calls']))
    _leak.sort()
    print('    leakage by FAM grade (Ye,B,R) -> share of grid calls:')
    for fam, probe, ye, b, r, share in _leak:
        print(f'      {code_tag(probe):<5} ({ye},{b},{r})  {share:6.3f}%  = {share / _per_unprinted:.2f}x the unprinted mean')
    _by_fam = {}
    for fam, probe, ye, b, r, share in _leak:
        _by_fam.setdefault(fam, []).append(share)
    _grade_means = [sum(v) / len(v) for _f, v in sorted(_by_fam.items())]
    _leak_monotonic = all((a <= b for a, b in zip(_grade_means, _grade_means[1:])))
    print(f'    mean share by FAM grade { {f: round(sum(v) / len(v), 3) for f, v in sorted(_by_fam.items())}} -> rises with FAM grade: {_leak_monotonic}')
    print(f"    printed codewords at or below that floor: {(', '.join((f'{code_tag(p)}:{PANEL[p]}' for p in _below)) if _below else 'none')}")
    _CLASS_COLOR = {'RNA': '#4C72B0', 'protein': '#C44E52', 'unprinted': '#999999'}
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))
    probes = sorted(PANEL)
    names = [display_name(p, PANEL[p]) for p in probes]
    heights = [per_probe[code_tag(p)]['n_spots'] for p in probes]
    colors = [_CLASS_COLOR[CLASSES[p]] for p in probes]
    labels_x = [f'{code_tag(p)}\n{n}' for p, n in zip(probes, names)]
    ax = axes[0]
    ax.bar(range(len(probes)), heights, color=colors)
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(labels_x, fontsize=6, rotation=90)
    ax.set_yscale('log')
    ax.set_ylabel('spots (log)')
    ax.axhline(_per_unprinted / 100.0 * _n_grid, color='k', ls=':', lw=1, label=f'unprinted-codeword floor ({_per_unprinted:.3f}% of calls)')
    ax.set_title(f'{RUN_ID} [{GENOTYPE}] - calls per probe  (n={len(called):,}; PR1-15 = G_layer 0, 16-30 = 1, 31 = pure Cy3)', fontsize=10)
    handles = [plt.Line2D([], [], marker='s', ls='', color=c, label=k) for k, c in _CLASS_COLOR.items()]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=7, ncol=4, loc='upper right', framealpha=0.85)
    ax = axes[1]
    conf_med = [per_probe[code_tag(p)]['median_confidence'] or 0 for p in probes]
    ax.bar(range(len(probes)), conf_med, color=colors)
    ax.axhline(CONFIDENCE_THRESHOLD, color='red', ls='--', lw=1, label=f'threshold {CONFIDENCE_THRESHOLD}')
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(labels_x, fontsize=6, rotation=90)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('median top-1 posterior')
    ax.set_title(f'Decode confidence per probe (probe {PURE_G_PROBE} has no posterior)', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / 'gene_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    _all_xy = called[called['Y'].notna()]
    y_max = float(_all_xy['Y'].max()) if len(_all_xy) else 1.0
    x_max = float(_all_xy['X'].max()) if len(_all_xy) else 1.0
    y_edges = np.arange(0, y_max + BIN_PX, BIN_PX)
    x_edges = np.arange(0, x_max + BIN_PX, BIN_PX)

    def _density(sub):
        h, _, _ = np.histogram2d(sub['Y'], sub['X'], bins=[y_edges, x_edges])
        return h
    n_panels = len(PANEL) + 1
    n_cols = 7
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for i, probe in enumerate(sorted(PANEL)):
        sub = _all_xy[_all_xy['probe'] == probe]
        dens = _density(sub)
        ax = axes[i]
        ax.imshow(dens, cmap='magma', interpolation='nearest', vmax=np.percentile(dens[dens > 0], 99) if (dens > 0).any() else 1)
        ax.set_title(f'{display_name(probe, PANEL[probe])}  ({code_tag(probe)}, n={len(sub):,})', fontsize=8, color='grey' if CLASSES[probe] == 'unprinted' else 'black')
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[len(PANEL)]
    dens_all = _density(_all_xy)
    ax.imshow(dens_all, cmap='gray_r', interpolation='nearest', vmax=np.percentile(dens_all[dens_all > 0], 99) if (dens_all > 0).any() else 1)
    ax.set_title(f'all calls (n={len(_all_xy):,})', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for j in range(n_panels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(figure_dir / 'spatial_genes.png', dpi=200, bbox_inches='tight')
    plt.close()
    out_cols = [c for c in ('index', 'Y', 'X', 'gene', 'probe', 'label', 'confidence', 'G_layer', 'sum', 'R', 'Ye', 'G', 'B') if c in called.columns]
    called[out_cols].rename(columns={'gene': 'Gene', 'confidence': 'Probability'}).to_csv(read_dir / 'mapping_gmm.csv', index=False)
    print(f'\n  wrote mapping_gmm.csv ({len(called):,} calls)')
    _counts = np.array([per_probe[code_tag(p)]['n_spots'] for p in probes], dtype=float)
    _share = _counts / max(_counts.sum(), 1)
    summary = {'RUN_ID': RUN_ID, 'genotype': GENOTYPE, 'panel': params['PRISM_PANEL'], 'n_probes': len(PANEL), 'n_printed': sum((c != 'unprinted' for c in CLASSES.values())), 'SG': params.get('SG'), 'SUM_THRESHOLD': params.get('SUM_THRESHOLD'), 'G_THRESHOLD': params.get('G_THRESHOLD'), 'PURE_G_RATIO': params.get('PURE_G_RATIO'), 'G_layer_cuts': params.get('G_layer_cuts'), 'confidence_threshold': CONFIDENCE_THRESHOLD, 'n_detected': params.get('n_detected'), 'n_colour_path': n_pre, 'n_assigned': n_called, 'n_confident': n_conf, 'n_pure_g': int(len(pure_g)), 'n_calls_total': int(len(called)), 'unprinted_probes': [int(p) for p in _unprinted], 'unprinted_calls': _unprinted_n, 'unprinted_share_pct': round(_unprinted_share, 3), 'per_unprinted_share_pct': round(_per_unprinted, 4), 'printed_at_or_below_unprinted_floor': [f'{code_tag(p)}:{PANEL[p]}' for p in _below], 'leakage_by_fam_grade': [{'probe': int(pr), 'grades_ye_b_r': [int(ye), int(b), int(r)], 'pct_of_grid_calls': sh} for _f, pr, ye, b, r, sh in _leak], 'leakage_rises_with_fam_grade': bool(_leak_monotonic), 'leakage_mean_share_by_fam_grade': {int(f): round(sum(v) / len(v), 4) for f, v in sorted(_by_fam.items())}, 'top_probe_share_pct': round(100.0 * float(_share.max()), 2), 'top_probe': code_tag(probes[int(_share.argmax())]), 'per_probe': per_probe}
    with open(read_dir / 'calling_summary.yaml', 'w') as f:
        summary['unprinted_probes_all'] = [p for p, c in CLASSES.items() if c == 'unprinted']
        summary['protein_specificity_validated'] = False
        yaml.safe_dump(plain(summary), f, sort_keys=False)
    print(f"  wrote calling_summary.yaml (top probe {summary['top_probe']} = {summary['top_probe_share_pct']}% of calls)")

    from prism.qc import generate_gene_calling_qc
    from prism.qc.reports import image_metadata
    shape, pixel_size = image_metadata(context.run_dir / 'stitched')
    generate_gene_calling_qc(
        called[out_cols], read_dir, RUN_ID,
        features=[PANEL[p] for p in sorted(PANEL)],
        excluded_features=[PANEL[p] for p in PANEL if CLASSES[p] == 'unprinted'],
        detected_spots=params.get('n_detected'), img_shape=shape,
        pixel_size_um=pixel_size,
    )
