"""PR31 production profile contracts: physical slots, algorithm parity and resume."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from prism.pipeline31 import main
from prism.gene_calling.prism31.context import load_context
from prism.gene_calling.prism31.codebook import PROBE_GRADES,build_label_to_probe

def fixture_data(per_probe=240):
 rng=np.random.default_rng(0);rows=[]
 for probe,(rg,yg,gg,bg) in PROBE_GRADES.items():
  scale=rng.normal(1.,.06,per_probe).clip(.6,1.6)
  r=4000*rg/4*scale+40; y=4000*yg/4*scale+40; bi=4000*bg/4*scale+40; gi=2500*gg*scale+40
  total=r+y+bi;g=gi*np.exp(-.6*y/total);b=bi*np.exp(-.05*3*gi/total)+.25*g
  rows.append(pd.DataFrame({'truth':probe,'r01_ex628':r,'r01_ex586':y,'r01_ex531':g,'r01_ex485':b}))
 rows.append(pd.DataFrame({'truth':np.zeros(3000,dtype=int),**{c:rng.uniform(5,90,3000) for c in ('r01_ex628','r01_ex586','r01_ex531','r01_ex485')}}))
 f=pd.concat(rows,ignore_index=True);f['index']=np.arange(len(f));f['Y']=rng.integers(0,4000,len(f));f['X']=rng.integers(0,4000,len(f));return f

def prepare(base):
 base.mkdir(parents=True,exist_ok=True);rid='SYNTHETIC_PR31';rd=base/(rid+'_processed')/'readout';rd.mkdir(parents=True)
 (rd/'readout_params.yaml').write_text('fixture: synthetic\\n');f=fixture_data();f[['index','Y','X']].to_csv(rd/'position.csv',index=False);f[['index','r01_ex628','r01_ex586','r01_ex531','r01_ex485']].to_csv(rd/'intensity.csv',index=False)
 pd.DataFrame([{'probe':p,'gene':f'Target{p}','class':'RNA'} for p in range(1,32)]).to_csv(base/'panel.csv',index=False)
 (base/'readout.yaml').write_text(yaml.safe_dump({'base_dir':str(base)}))
 cfg={'schema':'prism_run_v1','profile':'prism31_legacy_v1','run_id':rid,'base_dir':str(base),'panel':'panel.csv','panel_id':'synthetic31','readout_config_dir':'.','calling':{}}
 path=base/'run.yaml';path.write_text(yaml.safe_dump(cfg));return path,rd,f

@pytest.fixture(scope='module')
def decoded(tmp_path_factory):
 cfg,rd,truth=prepare(tmp_path_factory.mktemp('pr31'))
 for stage in ('preprocess','gmm','annotate','validate'): assert main(['--config',str(cfg),'--stage',stage])==0
 return cfg,rd,truth

def test_physical_slots_and_full_chain(decoded):
 cfg,rd,truth=decoded
 assert PROBE_GRADES[31]==(0,0,1,0) and build_label_to_probe()=={p:p for p in range(1,31)}
 params=yaml.safe_load((rd/'params.yaml').read_text());assert params['GLAYER']==2 and params['PURE_G_PROBE']==31 and params['SG']==3 and params['PURE_G_RATIO']==1
 calls=pd.read_csv(rd/'spots_called.csv').set_index('index');expected=truth.set_index('index').truth.reindex(calls.index);real=expected!=0
 assert (calls.loc[real,'probe']==expected[real]).mean()>.94
 assert (calls.probe==31).sum()>200
 counts=pd.read_csv(rd/'gene_counts.csv');assert len(counts)==31 and counts.n_spots.sum()==len(calls)
 result=json.loads((rd/'validation.json').read_text());assert result['profile']=='prism31_legacy_v1' and result['printed_slots']==31
 for row in (1,2):
  for layer in (1,2):assert (rd/f'figures/{row}-layer{layer}.png').is_file()
 assert not (rd/'figures/1-layer3.png').exists()

def test_unchanged_resume_and_panel_drift(decoded):
 cfg,rd,_=decoded;stamp=(rd/'intensity_preprocessed.csv').stat().st_mtime_ns
 main(['--config',str(cfg),'--stage','preprocess','--resume']);assert stamp==(rd/'intensity_preprocessed.csv').stat().st_mtime_ns
 panel=cfg.parent/'panel.csv';raw=panel.read_bytes()
 try:
  panel.write_bytes(raw.replace(b'Target1,',b'Changed1,'))
  with pytest.raises(ValueError,match='unchanged resume'): main(['--config',str(cfg),'--stage','preprocess','--resume'])
 finally:panel.write_bytes(raw)

def test_reject_wrong_panel_scope(tmp_path):
 cfg,rd,_=prepare(tmp_path);table=pd.read_csv(tmp_path/'panel.csv');table.iloc[:-1].to_csv(tmp_path/'panel.csv',index=False)
 with pytest.raises(ValueError,match='each PR31 slot'):load_context(cfg)

def test_legacy_preprocessing_numerical_parity(decoded,tmp_path):
 legacy=os.environ.get('PR31_LEGACY_SOURCE')
 if not legacy:pytest.skip('Set explicit retained legacy source for parity qualification')
 cfg,rd,_=decoded;legacy_base=tmp_path/'legacy';_,old_rd,_=prepare(legacy_base)
 env={**os.environ,'PRISM_BASE_DIR':str(legacy_base),'PYTHONNOUSERSITE':'1'}
 result=subprocess.run([sys.executable,'-B',str(Path(legacy)/'gene_calling_1_preprocess.py'),'SYNTHETIC_PR31'],cwd='C:/',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
 assert result.returncode==0,result.stdout.decode(errors='replace')[-4000:]
 for name in ('intensity_preprocessed.csv','intensity_G.csv'):
  a=pd.read_csv(rd/name);b=pd.read_csv(old_rd/name);pd.testing.assert_frame_equal(a.drop(columns=['G_layer'],errors='ignore'),b.drop(columns=['G_layer'],errors='ignore'),rtol=1e-12,atol=1e-12)
 a=yaml.safe_load((rd/'params.yaml').read_text());b=yaml.unsafe_load((old_rd/'params.yaml').read_text())
 for key in ('SG','PURE_G_RATIO','SUM_THRESHOLD','G_THRESHOLD','colour_maxima','LABEL_TO_PROBE'):
  assert a[key]==b[key]

