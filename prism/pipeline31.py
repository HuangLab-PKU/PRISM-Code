"""Shared post-stitch run pipeline: python -m prism.pipeline --config RUN.yaml."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback

from .gene_calling.prism31.context import load_context

STAGES = ('readout', 'preprocess', 'gmm', 'annotate', 'validate')
OUTPUTS = {
    'readout': ('position.csv', 'intensity.csv', 'readout_params.yaml',
                'readout_qc.png', 'readout_qc.json'),
    'preprocess': ('params.yaml', 'intensity_preprocessed.csv', 'intensity_G.csv'),
    'gmm': ('intensity_labeled.csv', 'figures/layer_figures.yaml',
            *tuple(f'figures/{row}-layer{k}.png' for row in (1, 2) for k in (1, 2))),
    'annotate': ('mapping_gmm.csv', 'gene_counts.csv', 'calling_summary.yaml',
                 'gene_calling_qc.png', 'gene_calling_qc.json'),
    'validate': ('validation.json',),
}


INTERMEDIATE = ('intensity_preprocessed.csv', 'intensity_labeled.csv', 'intensity_G.csv', 'params.yaml')


def product(ctx, name):
    return (ctx.work_dir if name in INTERMEDIATE else ctx.read_dir) / name


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path):
    st = path.stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def identity(ctx, stage):
    package = Path(__file__).parent
    code = {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))}
    required = {'readout': [], 'preprocess': ['position.csv', 'intensity.csv'],
                'gmm': list(OUTPUTS['preprocess']),
                'annotate': ['position.csv', 'params.yaml', 'intensity_labeled.csv', 'intensity_G.csv'],
                'validate': [n for s in STAGES[:-1] for n in OUTPUTS[s]]}[stage]
    source = {name: pin(product(ctx, name)) for name in required}
    if stage == 'readout':
        source['stitched'] = {p.name: pin(p) for p in sorted((ctx.run_dir / 'stitched').glob('*.tif'))}
        if not source['stitched'] and not (ctx.run_dir / 'stitched/mosaic.ome.zarr').is_dir():
            raise FileNotFoundError('No stitched image input')
        if not source['stitched']:
            raise ValueError('Finalize the mosaic to OME-TIFF before a resumable production readout')
    return {'run_id': ctx.run_id, 'profile': ctx.config['profile'],
            'config_sha256': digest(ctx.config_path), 'panel_sha256': digest(ctx.panel_path),
            'readout_config_sha256': digest(ctx.readout_config_dir / 'readout.yaml'),
            'code': code, 'inputs': source,
            'readout_origin': 'shared_pipeline' if (ctx.read_dir / 'pipeline_records/readout.json').exists() or stage == 'readout' else 'external_tables'}


def validate_readout(ctx):
    import numpy as np
    import pandas as pd
    p = pd.read_csv(ctx.read_dir / 'position.csv')
    i = pd.read_csv(ctx.read_dir / 'intensity.csv')
    if len(p) == 0 or not p['index'].is_unique or not i['index'].is_unique:
        raise ValueError('Empty or duplicate readout identities')
    if not np.array_equal(p['index'], i['index']):
        raise ValueError('Position/intensity IDs differ; no silent intersection is allowed')
    if not np.isfinite(p[['Y', 'X']].to_numpy()).all() or not np.isfinite(i.to_numpy()).all():
        raise ValueError('Nonfinite readout values')
    return p


def validate(ctx, audit=False):
    import numpy as np
    import pandas as pd
    from PIL import Image
    import yaml
    from prism.readout import mosaic as mosaic_io
    from prism.qc import require_qc_artifacts
    require_qc_artifacts(ctx.read_dir, 'readout')
    require_qc_artifacts(ctx.read_dir, 'gene_calling')
    if not audit:
        # Normal execution already builds these outputs from matched in-memory data.
        # Check only small handoff metadata; do not read millions of rows back again.
        summary = yaml.safe_load((ctx.read_dir / 'calling_summary.yaml').read_text())
        meta = yaml.safe_load((ctx.read_dir / 'figures/layer_figures.yaml').read_text())
        printed = sum(c != 'unprinted' for c in ctx.classes.values())
        if summary['n_printed'] != printed or summary['n_probes'] != 31:
            raise ValueError('Output summary and configured panel differ')
        if sum(v['n_spots'] for v in summary['per_probe'].values()) != summary['n_calls_total']:
            raise ValueError('Output summary count conservation failed')
        if (meta['width_px'], meta['height_px']) != (800, 700):
            raise ValueError('Manual-gate image metadata dimensions differ')
        for row in (1, 2):
            for k in (1, 2):
                if not (ctx.read_dir / f'figures/{row}-layer{k}.png').is_file():
                    raise FileNotFoundError('Missing manual-gate image')
        save(ctx.read_dir / 'validation.json', {
            'status': 'ROUTINE_OUTPUT_CHECKS_PASSED', 'run_id': ctx.run_id,
            'validation_mode': 'routine', 'tables_reread': False,
            'detected_spots': summary['n_detected'], 'called_spots': summary['n_calls_total'],
            'printed_slots': printed, 'manual_gate_size': [800, 700],
            'unprinted_probes_all': [p for p,c in ctx.classes.items() if c == 'unprinted'],
            'profile': 'prism31_legacy_v1', 'biological_specificity_validated': False})
        return
    positions = validate_readout(ctx)
    calls = pd.read_csv(ctx.read_dir / 'mapping_gmm.csv', keep_default_na=False).rename(columns={'Gene': 'gene', 'Probability': 'confidence'})
    counts = pd.read_csv(ctx.read_dir / 'gene_counts.csv', keep_default_na=False)
    if not calls['index'].is_unique or not calls['index'].isin(positions['index']).all():
        raise ValueError('Calls have duplicate or unknown identities')
    xy = calls[['Y', 'X']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite called coordinates')
    h, w = mosaic_io.mosaic_shape(ctx.run_dir / 'stitched', 1, 'cy3')
    if not calls.Y.between(0, h - 1).all() or not calls.X.between(0, w - 1).all():
        raise ValueError('Called coordinates outside mosaic')
    if len(counts) != 31 or not counts.probe.is_unique or int(counts.n_spots.sum()) != len(calls):
        raise ValueError('Count conservation failed')
    for probe in range(1, 32):
        row = counts.loc[counts.probe == probe].iloc[0]
        if row['gene'] != ctx.panel[probe] or row['class'] != ctx.classes[probe]:
            raise ValueError(f'Panel annotation differs at slot {probe}')
        subset = calls[calls.probe == probe]
        if not (subset.gene == ctx.panel[probe]).all() or len(subset) != row.n_spots:
            raise ValueError(f'Call/count annotation differs at slot {probe}')
    for row in (1, 2):
        for k in (1, 2):
            with Image.open(ctx.read_dir / f'figures/{row}-layer{k}.png') as image:
                if image.size != (800, 700):
                    raise ValueError('Manual-gate image dimensions differ')
    meta = yaml.safe_load((ctx.read_dir / 'figures/layer_figures.yaml').read_text())
    if meta['width_px'] != 800 or meta['height_px'] != 700 or not np.isclose(meta['px_size'], .002):
        raise ValueError('Manual-gate coordinate mapping differs')
    summary = yaml.safe_load((ctx.read_dir / 'calling_summary.yaml').read_text())
    printed = sum(c != 'unprinted' for c in ctx.classes.values())
    if summary['n_printed'] != printed:
        raise ValueError('Printed target count differs')
    save(ctx.read_dir / 'validation.json', {
        'status': 'TECHNICAL_PASS_VISUAL_REVIEW_PENDING', 'run_id': ctx.run_id,
        'detected_spots': len(positions), 'called_spots': len(calls),
        'printed_slots': printed, 'unprinted_probes_all': [p for p,c in ctx.classes.items() if c == 'unprinted'],
        'profile': 'prism31_legacy_v1', 'manual_gate_size': [800, 700],
        'biological_specificity_validated': False})


def require_predecessors(ctx, stage, audit=False):
    """Reject combining stage outputs made under different recipes or inputs."""
    for prior in STAGES[:STAGES.index(stage)]:
        receipt = ctx.read_dir / 'pipeline_records' / f'{prior}.json'
        # Explicit calling-only entry may import readout tables from another producer.
        # The preprocess receipt binds their actual identities from that point onward.
        if prior == 'readout' and not receipt.exists():
            continue
        if not receipt.exists():
            raise ValueError(f'{stage}: missing predecessor receipt for {prior}')
        previous = json.loads(receipt.read_text())
        if previous.get('status') != 'PASS':
            raise ValueError(f'{stage}: predecessor {prior} is not successful')
        current_recipe = {
            'run_id': ctx.run_id, 'profile': ctx.config['profile'],
            'config_sha256': digest(ctx.config_path), 'panel_sha256': digest(ctx.panel_path),
            'readout_config_sha256': digest(ctx.readout_config_dir / 'readout.yaml')}
        if any(previous['identity'].get(k) != v for k, v in current_recipe.items()):
            raise ValueError(f'{stage}: predecessor {prior} recipe drift')
        for name in OUTPUTS[prior]:
            if not (product(ctx, name)).is_file():
                raise FileNotFoundError(f'Missing predecessor output: {name}')
        if audit:
            outputs = {name: pin(product(ctx, name)) for name in OUTPUTS[prior]}
            if previous['identity'] != identity(ctx, prior) or previous['outputs'] != outputs:
                raise ValueError(f'{stage}: predecessor {prior} recipe/input/output drift')


def execute(ctx, stage, resume=False, audit=False):
    ctx.read_dir.mkdir(parents=True, exist_ok=True)
    records = ctx.read_dir / 'pipeline_records'
    records.mkdir(exist_ok=True)
    receipt = records / f'{stage}.json'
    require_predecessors(ctx, stage, audit=audit)
    if stage == 'preprocess':
        from prism.qc.reports import ensure_readout_qc
        ensure_readout_qc(ctx.read_dir, ctx.run_id)
    before = identity(ctx, stage)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        current = {name: pin(product(ctx, name)) for name in OUTPUTS[stage]}
        if not resume or previous['identity'] != before or previous['outputs'] != current:
            raise ValueError(f'{stage}: existing receipt requires exact unchanged resume')
        print(f'SKIP verified {stage}', flush=True)
        return
    if any((product(ctx, name)).exists() for name in OUTPUTS[stage]):
        raise ValueError(f'{stage}: partial/unowned outputs require explicit recovery; refusing overwrite')
    print(f'START {ctx.run_id} {stage}', flush=True)
    started = time.time()
    save(records / f'{stage}.attempt_{time.time_ns()}.json',
         {'stage': stage, 'started': started, 'identity_before': before})
    if stage == 'readout':
        from .readout import pipeline
        from .readout.spot_detection import _get_spotiflow_model
        model_dir = Path(ctx.config['detector_model_dir'])
        if not model_dir.is_dir():
            raise FileNotFoundError('Local Spotiflow model unavailable; downloads are not permitted')
        _get_spotiflow_model(model_path=model_dir)
        pipeline.load_config(ctx.readout_config_dir)
        if pipeline.DETECTION_MODEL_PATH != str(model_dir.resolve()):
            raise ValueError('Run/readout detector model paths differ')
        if pipeline.BASE_DIR.resolve() != ctx.run_dir.parent.resolve():
            raise ValueError('Readout and run base_dir differ')
        pipeline.main(ctx.run_id, ctx.run_dir / 'stitched', ctx.read_dir)
        if audit:
            validate_readout(ctx)
    elif stage == 'preprocess':
        if audit:
            validate_readout(ctx)
        from .gene_calling.prism31.preprocess import run
        run(ctx)
    elif stage == 'gmm':
        from .gene_calling.prism46.gmm import run
        run(ctx)
    elif stage == 'annotate':
        from .gene_calling.prism31.annotate import run
        run(ctx)
    elif stage == 'validate':
        validate(ctx, audit=audit)
        from .density_delivery import publish_density
        publish_density(ctx.read_dir, 'mapping_gmm.csv', [ctx.panel[p] for p in ctx.panel if ctx.classes[p] != 'unprinted'], ctx.run_id)
    else:
        raise ValueError(stage)
    if audit:
        after = identity(ctx, stage)
        if before != after:
            save(records / f'{stage}.identity_mismatch_{time.time_ns()}.json',
                 {'stage': stage, 'identity_before': before, 'identity_after': after,
                  'different_fields': [k for k in before if before[k] != after[k]]})
            raise RuntimeError('Source/config/input changed during stage; outputs retained without acceptance')
    if stage in ('readout', 'annotate'):
        from prism.qc import require_qc_artifacts
        require_qc_artifacts(ctx.read_dir, 'readout' if stage == 'readout' else 'gene_calling')
    outputs = {name: pin(product(ctx, name)) for name in OUTPUTS[stage]}
    save(receipt, {'stage': stage, 'identity': before, 'outputs': outputs,
                   'started': started, 'finished': time.time(), 'status': 'PASS',
                   'validation_mode': 'audit' if audit else 'routine'})
    print(f'DONE {ctx.run_id} {stage}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run-id', help='Assert the explicit run identity (also visible to live-reader checks)')
    ap.add_argument('--stage', choices=(*STAGES, 'all', 'calling'), default='all')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--audit', action='store_true',
                    help='Opt in to full-table readback and before/after source comparison')
    args = ap.parse_args(argv)
    ctx = load_context(args.config)
    if args.run_id is not None and args.run_id != ctx.run_id:
        raise ValueError('CLI/config run_id differ')
    ctx.read_dir.mkdir(parents=True, exist_ok=True)
    lock = ctx.read_dir / 'pipeline.lock'
    with lock.open('x', encoding='utf8') as f:
        json.dump({'pid': os.getpid(), 'run_id': ctx.run_id, 'started': time.time()}, f)
    try:
        stages = STAGES if args.stage == 'all' else (STAGES[1:] if args.stage == 'calling' else (args.stage,))
        for stage in stages:
            execute(ctx, stage, args.resume, audit=args.audit)
    except BaseException:
        save(ctx.read_dir / f'pipeline_failure_{time.time_ns()}.json',
             {'stage': args.stage, 'error': traceback.format_exc()})
        raise
    finally:
        owner = json.loads(lock.read_text())
        if owner['pid'] != os.getpid():
            raise RuntimeError('Pipeline lock ownership changed')
        lock.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
