"""Explicit run inputs for the PRISM46 v1 profile; no filename inference."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def plain(value):
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class RunContext:
    config_path: Path
    config: dict
    run_id: str
    run_dir: Path
    read_dir: Path
    panel_path: Path
    panel_id: str
    panel: dict
    classes: dict
    options: dict
    readout_config_dir: Path


def load_context(config_path):
    path = Path(config_path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict) or cfg.get('schema') != 'prism_run_v1':
        raise ValueError('Expected schema prism_run_v1')
    if cfg.get('profile') != 'prism46_v1':
        raise ValueError('This run driver currently supports explicit prism46_v1 only')
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else (path.parent / p).resolve()
    rid = cfg['run_id']
    if not isinstance(rid, str) or not rid or '/' in rid or '\\' in rid:
        raise ValueError('run_id must be a nonempty filename component')
    panel_path = resolve(cfg['panel'])
    table = pd.read_csv(panel_path, keep_default_na=False)
    if set(table.columns) != {'probe', 'gene', 'class'}:
        raise ValueError('Panel columns must be probe,gene,class')
    if table['probe'].duplicated().any() or set(table['probe']) != set(range(1, 47)):
        raise ValueError('Panel must contain each PRISM46 slot exactly once')
    if not table['class'].isin(['RNA', 'protein', 'unprinted']).all():
        raise ValueError('Unknown panel class')
    for row in table.itertuples():
        if not row.gene:
            raise ValueError('Empty target name')
    if not ((table['class'] == 'unprinted') == (table['gene'] == '-')).all():
        raise ValueError('Unprinted slots must have gene -, printed slots must have a name')
    options = cfg.get('calling', {})
    if set(options) - {'sg', 'pure_g_ratio', 'confidence_threshold', 'bin_px'}:
        raise ValueError('Unsupported calling override')
    for key in ('sg', 'pure_g_ratio', 'bin_px'):
        if key in options and (not np.isfinite(options[key]) or options[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= options.get('confidence_threshold', .8) <= 1:
        raise ValueError('confidence_threshold must be in [0, 1]')
    run_dir = resolve(cfg['base_dir']) / f'{rid}_processed'
    return RunContext(path, cfg, rid, run_dir, run_dir / 'readout', panel_path,
                      str(cfg.get('panel_id', 'configured_panel')),
                      dict(zip(table.probe, table.gene)),
                      dict(zip(table.probe, table['class'])), options,
                      resolve(cfg['readout_config_dir']))
