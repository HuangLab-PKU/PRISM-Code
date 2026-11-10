"""Shared QC for single-acquisition and colour-grid profiles; no SPRINTseq dependency."""
from __future__ import annotations

from collections import Counter
from itertools import zip_longest
from pathlib import Path
from functools import wraps
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

CHUNK = 500_000
SAMPLE = 200_000
FrameSource = pd.DataFrame | str | Path
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']


def _chunks(source):
    if isinstance(source, pd.DataFrame):
        for start in range(0, max(1, len(source)), CHUNK):
            yield source.iloc[start:start + CHUNK]
    else:
        yield from pd.read_csv(source, chunksize=CHUNK, keep_default_na=False, low_memory=False)


def _identities(sources):
    return {str(Path(p)): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
            for p in sources if not isinstance(p, pd.DataFrame)}


def _verify_sources(identities):
    if _identities(identities) != identities:
        raise ValueError('QC input changed during aggregation')


def _numbers(series):
    return pd.to_numeric(series.replace('', np.nan), errors='raise').to_numpy(float)


def _column(frame, candidates):
    return next((name for name in candidates if name in frame), None)


class _Sample:
    """Deterministic bottom-hash rows; uniform coverage without sorted-prefix bias."""
    def __init__(self):
        self.seen = 0
        self.keys = np.empty(0, np.uint64)
        self.rows = pd.DataFrame()

    def add(self, frame):
        ids = np.arange(self.seen, self.seen + len(frame), dtype=np.uint64)
        self.seen += len(frame)
        # Odd multiplication permutes uint64 row IDs; overflow is intentional.
        keys = ids * np.uint64(11400714819323198485)
        keys = np.concatenate([self.keys, keys])
        rows = pd.concat([self.rows, frame], ignore_index=True)
        if len(keys) > SAMPLE:
            selected = np.argpartition(keys, SAMPLE - 1)[:SAMPLE]
            keys, rows = keys[selected], rows.iloc[selected].reset_index(drop=True)
        self.keys, self.rows = keys, rows

    def metadata(self):
        return {'method': 'deterministic bottom-hash over all source rows',
                'rows': len(self.rows), 'source_rows': self.seen,
                'bounded_rows': SAMPLE}


class _Spatial:
    def __init__(self, shape=None, bin_pixels=200):
        self.shape = shape
        self.bin = bin_pixels
        self.sparse = Counter()
        self.n = 0
        self.available = False

    def add(self, frame, mask=None):
        ycol = _column(frame, ['Y', 'y'])
        xcol = _column(frame, ['X', 'x'])
        if not ycol or not xcol:
            return
        self.available = True
        xy = frame[[ycol, xcol]].to_numpy(float)
        if not np.isfinite(xy).all() or (xy < 0).any():
            raise ValueError('Invalid QC spatial coordinates')
        if self.shape and ((xy[:, 0] >= self.shape[0]).any()
                           or (xy[:, 1] >= self.shape[1]).any()):
            raise ValueError('QC coordinates outside acquired image')
        if mask is not None:
            xy = xy[mask]
        self.n += len(xy)
        b = (xy // self.bin).astype(np.uint64)
        codes, counts = np.unique((b[:, 0] << np.uint64(32)) | b[:, 1],
                                  return_counts=True)
        self.sparse.update(dict(zip(codes.tolist(), counts.tolist())))

    def grid(self):
        if not self.available:
            return None
        height = max((key >> 32 for key in self.sparse), default=0) + 1
        width = max((key & 0xffffffff for key in self.sparse), default=0) + 1
        if self.shape:
            height, width = [int(np.ceil(n / self.bin)) for n in self.shape]
        if height * width > 16_000_000:
            raise ValueError('QC spatial grid too large; use larger bins')
        grid = np.zeros((height, width), np.int64)
        for key, count in self.sparse.items():
            grid[key >> 32, key & 0xffffffff] = count
        assert int(grid.sum()) == self.n
        return grid


def image_metadata(stitched):
    """Read only TIFF axes/calibration; leave unavailable units explicitly unknown."""
    import tifffile
    stitched = Path(stitched)
    paths = [stitched / 'mosaic.ome.tif'] + sorted(stitched.glob('cyc_*.tif'))
    for path in paths:
        if not path.is_file():
            continue
        with tifffile.TiffFile(path) as image:
            shape = tuple(image.series[0].shape[-2:])
            pixel = None
            if image.ome_metadata:
                node = ET.fromstring(image.ome_metadata).find('.//{*}Pixels')
                if node is not None:
                    x, y = node.get('PhysicalSizeX'), node.get('PhysicalSizeY')
                    units = [node.get('PhysicalSizeXUnit', 'um'),
                             node.get('PhysicalSizeYUnit', 'um')]
                    if x and y and all(u in ('um', 'µm', 'μm') for u in units):
                        if float(x) > 0 and np.isclose(float(x), float(y)):
                            pixel = float(x)
            return shape, pixel
    return None, None


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _style(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        plt = _setup()
        previous = set(plt.get_fignums())
        try:
            with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                                 'axes.spines.top': False, 'axes.spines.right': False}):
                return function(*args, **kwargs)
        finally:
            for number in set(plt.get_fignums()) - previous:
                plt.close(number)
    return wrapped


def _empty(ax, text):
    ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()


def _spatial_plot(fig, ax, grid, bin_pixels, pixel, title):
    from matplotlib.colors import LogNorm
    if grid is None or not grid.any():
        _empty(ax, 'No reported spatial points')
        return
    scale = pixel or 1.0
    extent = [0, grid.shape[1] * bin_pixels * scale,
              grid.shape[0] * bin_pixels * scale, 0]
    image = ax.imshow(np.ma.masked_equal(grid, 0), origin='upper', extent=extent,
                      cmap='viridis', norm=LogNorm(vmin=1, vmax=max(2, int(grid.max()))))
    units = 'um' if pixel else 'pixels'
    ax.set(title=title, xlabel=f'X ({units})', ylabel=f'Y ({units})')
    fig.colorbar(image, ax=ax, label=f'Spots / {bin_pixels * scale:g}-{units} bin',
                 shrink=.8)


def _publish(fig, report, output_dir, kind):
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    png = output / f'{kind}_qc.png'
    data = io.BytesIO()
    fig.savefig(data, format='png', dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    payload = data.getvalue()
    report.update(status='QC_COMPLETE', figure_sha256=hashlib.sha256(payload).hexdigest())
    tmp = png.with_suffix('.png.tmp')
    tmp.write_bytes(payload)
    tmp.replace(png)
    path = output / f'{kind}_qc.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
    require_qc_artifacts(output, kind)
    return report


def require_qc_artifacts(output_dir: str | Path, kind: str,
                         expected_rows: int | None = None) -> dict:
    """Completion requires a matching readable PNG/report pair, not a filename alone."""
    from PIL import Image
    root = Path(output_dir)
    png, metadata = root / f'{kind}_qc.png', root / f'{kind}_qc.json'
    report = json.loads(metadata.read_text(encoding='utf-8-sig'))
    if report['status'] != 'QC_COMPLETE':
        raise ValueError('QC report is not complete')
    if hashlib.sha256(png.read_bytes()).hexdigest() != report['figure_sha256']:
        raise ValueError('QC figure/report identity mismatch')
    with Image.open(png) as image:
        if min(image.size) < 100:
            raise ValueError('QC figure is empty or truncated')
        image.verify()
    if expected_rows is not None and report['input_rows'] != expected_rows:
        raise ValueError('QC row count differs from numerical handoff')
    return report


@_style
def generate_readout_qc(
    intensity: FrameSource, positions: FrameSource, output_dir: str | Path,
    run_id: str, channels: list[str], img_shape: tuple[int, int] | None = None,
    pixel_size_um: float | None = None, detection_counts: dict | None = None,
    exact_unique: int | None = None, acquired_cycles: list[int] | None = None,
) -> dict:
    """Exact full-row spatial/funnel/histogram QC; bounded sample for correlations."""
    if not channels or len(set(channels)) != len(channels):
        raise ValueError('Explicit distinct readout channels required')
    identities = _identities([intensity, positions])
    sample, spatial = _Sample(), _Spatial(img_shape)
    bins = np.linspace(0, np.log10(65536), 97)
    counts = np.zeros((len(channels), 96), np.int64)
    zero = np.zeros(len(channels), np.int64)
    overflow = np.zeros(len(channels), np.int64)
    total = 0
    for left, right in zip_longest(_chunks(intensity), _chunks(positions)):
        if left is None or right is None or len(left) != len(right):
            raise ValueError('QC position/intensity row mismatch')
        if 'index' in left and 'index' in right:
            if not np.array_equal(left['index'].to_numpy(), right['index'].to_numpy()):
                raise ValueError('QC position/intensity identities differ')
        values = left[channels].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Raw readout intensity must be finite and nonnegative')
        for i in range(len(channels)):
            counts[i] += np.histogram(np.log10(1 + np.minimum(values[:, i], 65535)), bins)[0]
        zero += (values == 0).sum(axis=0)
        overflow += (values > 65535).sum(axis=0)
        sample.add(pd.DataFrame(values, columns=channels))
        spatial.add(right)
        total += len(left)
    grid = spatial.grid()
    if grid is not None and int(grid.sum()) != total:
        raise ValueError('QC spatial count conservation failed')
    if not np.all(counts.sum(axis=1) == total):
        raise ValueError('QC intensity histogram count conservation failed')
    _verify_sources(identities)
    corr = np.full((len(channels), len(channels)), np.nan)
    if len(sample.rows) > 1:
        logged = np.log1p(sample.rows[channels].to_numpy(float))
        varying = np.flatnonzero(np.std(logged, axis=0) > 0)
        if len(varying) > 1:
            corr[np.ix_(varying, varying)] = np.corrcoef(logged[:, varying].T)
        elif len(varying) == 1:
            corr[varying[0], varying[0]] = 1.0
    report = {'run_id': run_id, 'input_rows': total, 'channels': list(channels),
              'acquired_cycles': acquired_cycles or [], 'source_identities': identities,
              'spatial_rows': spatial.n, 'bin_size_pixels': spatial.bin,
              'intensity_histogram_log10_1plus_edges': bins.tolist(),
              'intensity_histogram_counts': counts.tolist(),
              'zero_intensity_counts': zero.tolist(),
              'above_65535_counts': overflow.tolist(),
              'histogram_last_bin_includes_above_65535': True,
              'correlation_sample': sample.metadata(),
              'log1p_correlation': [[float(x) if np.isfinite(x) else None for x in row] for row in np.atleast_2d(corr)],
                  
              'funnel': {'per_channel_detections': detection_counts,
                         'exact_unique_coordinates': exact_unique,
                         'spatially_deduplicated': total},
              'limits': ['Readout QC does not validate biological specificity',
                         'Pooled channel correlations are not dye-crosstalk estimates']}
    plt = _setup()
    n = len(channels)
    fig = plt.figure(figsize=(max(12, n * 2.7), 8.5))
    top = fig.add_gridspec(1, 2, left=.07, right=.94, bottom=.44, top=.86,
                           wspace=.28, width_ratios=[1.15, 1])
    bottom = fig.add_gridspec(1, n, left=.07, right=.94, bottom=.10, top=.33,
                              wspace=.25)
    ax = fig.add_subplot(top[0, 0])
    _spatial_plot(fig, ax, grid, spatial.bin, pixel_size_um, 'All readout spots')
    ax = fig.add_subplot(top[0, 1])
    if len(sample.rows) > 1:
        image = ax.imshow(np.ma.masked_invalid(np.atleast_2d(corr)), vmin=-1,
                          vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(n), channels, rotation=45)
        ax.set_yticks(range(n), channels)
        ax.set_title(f'Log-intensity correlation ({len(sample.rows):,} sampled rows)')
        fig.colorbar(image, ax=ax, label='Pearson r', shrink=.8)
    else:
        _empty(ax, 'Correlation unavailable for fewer than two rows')
    ymax = max(.01, float((counts / max(total, 1)).max(initial=0)) * 1.08)
    for i, channel in enumerate(channels):
        ax = fig.add_subplot(bottom[0, i])
        ax.stairs(counts[i] / max(total, 1), bins, color=PALETTE[i % len(PALETTE)])
        ax.text(.98,
            .95,
            f'Zero: {zero[i] / max(total, 1):.1%}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9)
        ax.set(xlim=(bins[0], bins[-1]), ylim=(0, ymax), title=channel,
               xlabel='log10(1 + intensity)')
        ax.set_xticks([0, 1, 2, 3, 4])
        if overflow[i]:
            ax.text(.98,
                .81,
                f'Upper overflow: {overflow[i] / max(total, 1):.1%}',
                transform=ax.transAxes,
                ha='right',
                fontsize=8)
        if i == 0:
            ax.set_ylabel('Fraction of all spots / bin')
        else:
            ax.tick_params(labelleft=False)
    funnel = ''
    if detection_counts is not None:
        raw = sum(detection_counts.values())
        if raw < total or (exact_unique is not None and not raw >= exact_unique >= total):
            raise ValueError('Invalid readout funnel counts')
        funnel = f' | {raw:,} channel detections -> {exact_unique:,} exact -> {total:,} final' if exact_unique is not None else f' | {raw:,} detections -> {total:,} final'
    fig.suptitle(f'{run_id}: readout QC\n{funnel.lstrip(" |")}', fontsize=12)
    return _publish(fig, report, output_dir, 'readout')


@_style
def generate_gene_calling_qc(
    calls: FrameSource, output_dir: str | Path, run_id: str,
    diagnostics: dict | None = None, features: list[str] | None = None,
    excluded_features: tuple | list = (), detected_spots: int | None = None,
    img_shape: tuple[int, int] | None = None, pixel_size_um: float | None = None,
) -> dict:
    """SPRINTseq-like calling diagnostics without fabricated absent posteriors."""
    identities = _identities([calls])
    sample, spatial = _Sample(), _Spatial(img_shape)
    labels, q20 = Counter(), Counter()
    probability_bins = np.linspace(0, 1, 101)
    hist = np.zeros((2, 100), np.int64)
    thresholds = np.array([0, 3, 5, 10, 13, 20, 30, 40, 50])
    retained = np.zeros(len(thresholds), np.int64)
    excluded = {'background', 'infeasible', 'false_pos', 'unassigned_codeword', ''}
    excluded.update(str(x).lower() for x in excluded_features)
    total = target_n = valid_target = missing_confidence = 0
    source_label = None
    for frame in _chunks(calls):
        label = _column(frame, ['Gene', 'gene', 'top1_gene', 'predicted_label', 'label'])
        if label is None:
            raise ValueError('Calling QC requires real gene/feature labels')
        source_label = label
        names = frame[label].where(frame[label].notna(), '').astype(str)
        target = ~names.str.lower().isin(excluded).to_numpy()
        pcol = _column(frame,
            ['Probability',
            'top1_prob',
            'prediction_confidence',
            'confidence',
            'Confidence',
            'probability'])
        prob = _numbers(frame[pcol]) if pcol else np.full(len(frame), np.nan)
        if np.isinf(prob).any() or ((prob[np.isfinite(prob)] < 0) | (prob[np.isfinite(prob)] > 1)).any():
            raise ValueError('Invalid calling posterior/confidence')
        ecol = _column(frame, ['Entropy', 'entropy'])
        entropy = _numbers(frame[ecol]) if ecol else np.full(len(frame), np.nan)
        p2col = _column(frame, ['Probability_2', 'top2_prob', 'probability_2'])
        second = _numbers(frame[p2col]) if p2col else np.full(len(frame), np.nan)
        finite = np.isfinite(prob)
        valid = target & finite
        missing_confidence += int((~finite).sum())
        labels.update(names.value_counts().to_dict())
        q20.update(names[valid & (prob >= .99)].value_counts().to_dict())
        for i, mask in enumerate([valid, (~target) & finite]):
            hist[i] += np.histogram(prob[mask], probability_bins)[0]
        cutoffs = 1 - 10.0 ** (-thresholds / 10.0)
        retained += (prob[valid, None] >= cutoffs[None, :]).sum(axis=0)
        valid_target += int(valid.sum())
        spatial.add(frame, target)
        sample.add(pd.DataFrame({'Probability': prob, 'Entropy': entropy,
                                 'Margin': prob - second}))
        total += len(frame)
        target_n += int(target.sum())
    _verify_sources(identities)
    grid = spatial.grid()
    if detected_spots is not None and detected_spots < total:
        raise ValueError('Calling rows exceed reported readout count')
    trace = (diagnostics or {}).get('losses')
    losses = [] if trace is None else np.asarray(trace, float).ravel().tolist()
    if losses and not np.isfinite(losses).all():
        raise ValueError('Invalid fitting objective trace')
    all_features = list(dict.fromkeys([*(features or []), *labels]))
    per_feature = [{'feature': str(g), 'count': int(labels.get(g, 0)),
                    'n_posterior_ge_0_99': int(q20.get(g, 0)),
                    'target': str(g).lower() not in excluded} for g in all_features]
    report = {'run_id': run_id, 'input_rows': total, 'source_label_column': source_label,
              'detected_spots': detected_spots, 'assigned_target_rows': target_n,
              'reported_non_target_rows': total - target_n,
              'not_emitted_rows': None if detected_spots is None else detected_spots-total,
              'target_rows_with_reported_confidence': valid_target,
              'rows_without_reported_confidence': missing_confidence,
              'spatial_target_rows': spatial.n if spatial.available else None,
              'posterior_sample': sample.metadata(), 'source_identities': identities,
              'label_counts': dict(labels), 'per_feature': per_feature,
              'probability_edges': probability_bins.tolist(),
              'probability_histogram_target_and_non_target': hist.tolist(),
              'posterior_Q_thresholds': thresholds.tolist(),
              'targets_above_Q_threshold': retained.tolist(),
              'fitting_objective': [float(x) for x in losses],
              'missing_values_imputed': False,
              'limits': ['Posterior/Q scores are model diagnostics, not empirical error calibration',
                  
                         'Absent confidence, entropy or fitting traces are explicitly unavailable']}
    plt = _setup()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout='constrained')
    ax = axes[0, 0]
    categories = ['Targets', 'Reported non-targets']
    values = [target_n, total-target_n]
    if detected_spots is not None:
        categories.append('Not emitted')
        values.append(detected_spots-total)
    ax.barh(categories[::-1], values[::-1], color=PALETTE[0])
    ax.set(title='Calling outcomes', xlabel='Rows')
    ax = axes[0, 1]
    if hist.sum():
        for i, label in enumerate(['Target labels', 'Non-target labels']):
            ax.stairs(hist[i], probability_bins, label=label, color=PALETTE[i])
        ax.set(title='Reported posterior / confidence', xlabel='Probability', ylabel='Rows')
        ax.legend(fontsize=8)
    else:
        _empty(ax, 'Posterior/confidence not reported')
    posterior = sample.rows
    ent = posterior.Entropy.to_numpy() if len(posterior) else np.array([])
    ent = ent[np.isfinite(ent)]
    ax = axes[0, 2]
    if ent.size:
        ax.hist(ent, bins=60, color=PALETTE[0])
        ax.set(title='Entropy (sample)', xlabel='Entropy', ylabel='Sampled rows')
    else:
        _empty(ax, 'Entropy not reported')
    _spatial_plot(fig, axes[1, 0], grid, spatial.bin, pixel_size_um, 'Target-label spatial density')
    ax = axes[1, 1]
    if losses:
        ax.plot(np.arange(1, len(losses)+1), losses, color=PALETTE[0])
        ax.set(title='Reported fitting objective', xlabel='Iteration', ylabel='Objective')
    else:
        _empty(ax, 'Fitting objective trace not reported')
    top = sorted([x for x in per_feature if x['target']],
        key=lambda x:x['count'],
        reverse=True)[:20]
    ax = axes[1, 2]
    if top:
        y = np.arange(len(top))
        ax.barh(y, [x['count'] for x in top], color='#bdbdbd', label='All calls')
        if valid_target:ax.barh(y,
            [x['n_posterior_ge_0_99'] for x in top],
            color=PALETTE[0],
            label='Reported P >=0.99')
        ax.set_yticks(y, [x['feature'] for x in top], fontsize=7)
        ax.invert_yaxis()
        ax.set(title='Most abundant target labels', xlabel='Rows')
        ax.legend(fontsize=7)
    else:
        _empty(ax, 'No target labels')
    ax = axes[2, 0]
    if valid_target:
        ax.plot(thresholds, retained/valid_target, marker='o', color=PALETTE[0])
        ax.set(ylim=(0,
            1.05),
            title='Posterior-threshold retention',
            xlabel='Posterior-derived Q',
            ylabel='Fraction of targets with reported P')
    else:
        _empty(ax, 'No target posterior values')
    ax = axes[2, 1]
    if len(posterior):
        valid = np.isfinite(posterior.Probability) & np.isfinite(posterior.Entropy)
        if valid.any():
            from matplotlib.colors import LogNorm
            image = ax.hist2d(posterior.Probability[valid],
                posterior.Entropy[valid],
                bins=60,
                norm=LogNorm())
            fig.colorbar(image[3],ax=ax,shrink=.8)
            ax.set(title='Posterior vs entropy (sample)',xlabel='Probability',ylabel='Entropy')
        else:_empty(ax, 'Joint posterior / entropy not reported')
    else:_empty(ax, 'No calling rows')
    ax = axes[2, 2]
    margin = posterior.Margin.to_numpy() if len(posterior) else np.array([])
    margin = margin[np.isfinite(margin)]
    if margin.size:
        ax.hist(margin,bins=60,color=PALETTE[0])
        ax.set(title='Top-1 minus top-2 (sample)',xlabel='Posterior margin',ylabel='Sampled rows')
    else:_empty(ax,'Runner-up posterior not reported')
    fig.suptitle(f'{run_id}: gene-calling QC | {total:,} output rows; {missing_confidence:,} without reported confidence',
        fontsize=13)
    fig.supxlabel('Posterior/Q values are model diagnostics, not empirical error calibration.',
        fontsize=10)
    return _publish(fig, report, output_dir, 'gene_calling')


def ensure_readout_qc(
    read_dir: str | Path, run_id: str, intensity: FrameSource | None = None,
    positions: FrameSource | None = None, channels: list[str] | None = None,
    acquired_cycles: list[int] | None = None,
) -> dict:
    """Importing external raw tables must also obtain readout QC without refitting."""
    root = Path(read_dir)
    if (root / 'readout_qc.json').exists() and (root / 'readout_qc.png').exists():
        existing = json.loads((root / 'readout_qc.json').read_text(encoding='utf-8-sig'))
        if existing.get('status') == 'QC_COMPLETE':
            return require_qc_artifacts(root, 'readout')
        previous = root / 'qc_previous'
        previous.mkdir(exist_ok=True)
        for name in ('readout_qc.json', 'readout_qc.png'):
            target = previous / name
            if target.exists():
                raise FileExistsError('Previous nonstandard QC already preserved; inspect before replacing')
            (root / name).replace(target)
    intensity = root / 'intensity.csv' if intensity is None else intensity
    positions = root / 'position.csv' if positions is None else positions
    if channels is None:
        columns = intensity.columns if isinstance(intensity,
            pd.DataFrame) else pd.read_csv(intensity,
            nrows=0).columns
        channels = [name for name in columns if name != 'index']
    shape, pixel = image_metadata(root.parent / 'stitched')
    return generate_readout_qc(intensity, positions, root, run_id, channels,
                               img_shape=shape, pixel_size_um=pixel,
                               acquired_cycles=acquired_cycles)
