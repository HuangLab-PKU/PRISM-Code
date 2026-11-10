"""End-to-end contract for the shared PRISM46 profile."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from tifffile import imwrite
from prism.pipeline import main
from prism.gene_calling.prism46.codebook import PROBE_GRADES
from prism.gene_calling.prism46.context import load_context


def synthetic_readout(seed=0, per_probe=400):
    rng = np.random.default_rng(seed)
    rows = []
    for probe, (rgrade, ygrade, ggrade, bgrade) in PROBE_GRADES.items():
        scale = rng.normal(1., .06, per_probe).clip(.6, 1.6)
        r = 4000 * rgrade / 4 * scale + 40
        y = 4000 * ygrade / 4 * scale + 40
        bi = 4000 * bgrade / 4 * scale + 40
        gi = 2500 * ggrade * scale + 40
        total = r + y + bi
        g = gi * np.exp(-.6 * y / total)
        b = bi * np.exp(-.05 * 3 * gi / total) + .25 * g
        rows.append(pd.DataFrame({'probe_truth': probe, 'r01_ex628': r,
                                  'r01_ex586': y, 'r01_ex531': g, 'r01_ex485': b}))
    rows.append(pd.DataFrame({'probe_truth': np.zeros(3000, dtype=int),
                             **{c: rng.uniform(5, 90, 3000) for c in
                                ('r01_ex628','r01_ex586','r01_ex531','r01_ex485')}}))
    frame = pd.concat(rows, ignore_index=True)
    frame['index'] = np.arange(len(frame))
    frame['Y'] = rng.integers(0, 4000, len(frame))
    frame['X'] = rng.integers(0, 4000, len(frame))
    return frame


def prepare_run(base, run_id, omit46):
    base = Path(base);base.mkdir(parents=True, exist_ok=True)
    run = base / (run_id + '_processed');rd = run / 'readout';rd.mkdir(parents=True)
    (run / 'stitched').mkdir()
    frame = synthetic_readout()
    frame[['index','Y','X']].to_csv(rd/'position.csv', index=False)
    frame[['index','r01_ex628','r01_ex586','r01_ex531','r01_ex485']].to_csv(rd/'intensity.csv', index=False)
    (rd/'readout_params.yaml').write_text('fixture: synthetic\n')
    imwrite(run/'stitched/cyc_1_cy3.tif', np.zeros((4000,4000), dtype=np.uint16))
    panel=[]
    for p in range(1,47):
        absent=p in {2,10,12} or (omit46 and p==46)
        panel.append({'probe':p,'gene':'-' if absent else f'Target{p}',
                      'class':'unprinted' if absent else ('protein' if p in {1,5,9,46} else 'RNA')})
    pd.DataFrame(panel).to_csv(base/'panel.csv',index=False)
    (base/'readout.yaml').write_text(yaml.safe_dump({'base_dir':str(base)}))
    cfg={'schema':'prism_run_v1','profile':'prism46_v1','run_id':run_id,
         'base_dir':str(base),'panel':'panel.csv','panel_id':'explicit_fixture',
         'readout_config_dir':'.','calling':{'confidence_threshold':.91 if omit46 else .8}}
    path=base/'run.yaml';path.write_text(yaml.safe_dump(cfg))
    return path,rd,frame


@pytest.fixture(scope='module', params=[False, True])
def decoded(request, tmp_path_factory):
    config,rd,truth=prepare_run(tmp_path_factory.mktemp('shared46'), 'NO_GENOTYPE_IN_NAME', request.param)
    for stage in ('preprocess','gmm','annotate','validate'):
        assert main(['--config',str(config),'--stage',stage]) == 0
    return config,rd,truth,request.param


def test_full_shared_chain(decoded):
    config,rd,truth,omit46=decoded
    calls=pd.read_csv(rd/'spots_called.csv').set_index('index')
    expected=truth.set_index('index').probe_truth.reindex(calls.index)
    real=expected!=0
    assert (calls.loc[real,'probe']==expected[real]).mean() > .95
    assert (calls.probe==46).sum()>320
    counts=pd.read_csv(rd/'gene_counts.csv',keep_default_na=False)
    assert len(counts)==46 and counts.n_spots.sum()==len(calls)
    assert counts.loc[counts.probe==46,'class'].item()==('unprinted' if omit46 else 'protein')
    result=json.loads((rd/'validation.json').read_text())
    assert result['printed_slots']==(42 if omit46 else 43)
    assert result['manual_gate_size']==[800,700]


def test_resume_is_exact_and_rejects_panel_drift(decoded):
    config,rd,truth,omit46=decoded
    stamp=(rd/'intensity_preprocessed.csv').stat().st_mtime_ns
    main(['--config',str(config),'--stage','preprocess','--resume'])
    assert (rd/'intensity_preprocessed.csv').stat().st_mtime_ns==stamp
    panel=config.parent/'panel.csv';raw=panel.read_bytes()
    try:
        panel.write_bytes(raw.replace(b'Target1,',b'Changed1,'))
        with pytest.raises(ValueError, match='unchanged resume'):
            main(['--config',str(config),'--stage','preprocess','--resume'])
    finally:
        panel.write_bytes(raw)


def test_missing_or_duplicate_panel_slot_is_rejected(tmp_path):
    config,rd,frame=prepare_run(tmp_path,'FIXTURE',False)
    p=tmp_path/'panel.csv';table=pd.read_csv(p);table.iloc[:-1].to_csv(p,index=False)
    with pytest.raises(ValueError,match='each PRISM46 slot'):
        load_context(config)


def test_later_stage_rejects_prior_recipe_drift(decoded):
    config,rd,truth,omit46=decoded
    raw=config.read_bytes();cfg=yaml.safe_load(raw);cfg['calling']['sg']=4
    try:
        config.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError,match='predecessor preprocess'):
            main(['--config',str(config),'--stage','gmm'])
    finally:
        config.write_bytes(raw)


def test_confidence_override_reaches_saved_parameters(decoded):
    config,rd,truth,omit46=decoded
    expected=yaml.safe_load(config.read_text())['calling'].get('confidence_threshold',.8)
    assert yaml.safe_load((rd/'params.yaml').read_text())['confidence_threshold']==expected
    assert yaml.safe_load((rd/'calling_summary.yaml').read_text())['confidence_threshold']==expected



def test_source_change_records_both_identities_before_refusing(tmp_path, monkeypatch):
    from prism import pipeline as driver
    from prism.readout import pipeline as reader
    config,rd,frame=prepare_run(tmp_path,'DIAGNOSTIC',False)
    ctx=load_context(config)
    for name in driver.OUTPUTS['readout']:(rd/name).unlink()
    identities=iter([{'generation':'before'},{'generation':'after'}])
    monkeypatch.setattr(driver,'identity',lambda *a:next(identities))
    monkeypatch.setattr(reader,'load_config',lambda *a:None)
    monkeypatch.setattr(reader,'BASE_DIR',tmp_path)
    monkeypatch.setattr(reader,'main',lambda *a:None)
    monkeypatch.setattr(driver,'validate_readout',lambda *a:None)
    with pytest.raises(RuntimeError,match='changed during stage'):
        driver.execute(ctx,'readout',audit=True)
    records=rd/'pipeline_records'
    attempt=json.loads(next(records.glob('readout.attempt_*.json')).read_text())
    refusal=json.loads(next(records.glob('readout.identity_mismatch_*.json')).read_text())
    assert attempt['identity_before']=={'generation':'before'}
    assert refusal['identity_after']=={'generation':'after'}
    assert refusal['different_fields']==['generation']
    assert not (records/'readout.json').exists()


def test_routine_readout_does_not_repeat_readback_or_end_identity(tmp_path,monkeypatch):
    from prism import pipeline as driver
    from prism.readout import pipeline as reader
    config,rd,frame=prepare_run(tmp_path,'ROUTINE',False)
    ctx=load_context(config)
    for name in driver.OUTPUTS['readout']:(rd/name).unlink()
    calls=[]
    def identity(*args):
        calls.append(True)
        assert len(calls)==1,'Routine execution must not repeat the end identity audit'
        return {'recipe':'fixed'}
    def write(*args):
        for name in driver.OUTPUTS['readout']:(rd/name).write_text('fixture output')
    def forbidden(*args):raise AssertionError('Routine execution must not reread tables')
    monkeypatch.setattr(driver,'identity',identity)
    monkeypatch.setattr(reader,'load_config',lambda *a:None)
    monkeypatch.setattr(reader,'BASE_DIR',tmp_path)
    monkeypatch.setattr(reader,'main',write)
    monkeypatch.setattr(driver,'validate_readout',forbidden)
    driver.execute(ctx,'readout')
    receipt=json.loads((rd/'pipeline_records/readout.json').read_text())
    assert receipt['status']=='PASS' and receipt['validation_mode']=='routine'
