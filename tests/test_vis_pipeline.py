"""Configured VIS handoff keeps shuffled gene/word bindings and counts backgrounds."""
import json
import numpy as np,pandas as pd,pytest,yaml
from prism import vis_pipeline

def prepare(tmp_path):
 rid='SYNTHETIC_VIS';rd=tmp_path/(rid+'_processed')/'readout';rd.mkdir(parents=True)
 panel=pd.DataFrame([{'gene':f'Target{i}','code_id':f'Code{i}','bits':f'{i:05b}'} for i in range(1,32)]).sample(frac=1,random_state=9)
 panel.to_csv(tmp_path/'panel.csv',index=False)
 pd.DataFrame({'index':np.arange(40)+100,'Y':np.arange(40),'X':np.arange(40)}).to_csv(rd/'position.csv',index=False)
 pd.DataFrame({'index':np.arange(40)+100,**{ch:np.arange(40)+1 for ch in ['cy3','430','TxRed','FAM','cy5']}}).to_csv(rd/'intensity.csv',index=False)
 cfg={'run_id':rid,'base_dir':str(tmp_path),'panel':str(tmp_path/'panel.csv'),'channels':['cy3','430','TxRed','FAM','cy5'],'cycles':[3],'profile':'vis31_5bit_single_acquisition_v1'}
 path=tmp_path/'decode.yaml';path.write_text(yaml.safe_dump(cfg));return path,rd,cfg,panel

def test_shuffled_panel_and_background_conservation(tmp_path,monkeypatch):
 path,rd,cfg,panel=prepare(tmp_path)
 def fake(values,bits,genes,words,**kwargs):
  assert values.shape==(40,5,1)
  assert all(int(g.removeprefix('Target'))==int(w,2) for g,w in zip(genes,words))
  assert np.array_equal(bits[:,:,0],np.asarray([[int(x) for x in w] for w in words]))
  return pd.DataFrame({'index':np.arange(40),'Sequence':[words[0]]*30+['Background']*10,'Gene':[genes[0]]*30+['Background']*10,'Probability':np.ones(40)*.99,'Probability_2':np.ones(40)*.01,'Entropy':np.ones(40)*.1}),{'fixture':True}
 monkeypatch.setattr(vis_pipeline,'decode',fake)
 assert vis_pipeline.main(['--config',str(path),'--run-id',cfg['run_id']])==0
 result=pd.read_csv(rd/'mapped_genes.csv');assert result['index'].tolist()==list(range(100,140))
 assert result.Gene.iloc[0]==panel.gene.iloc[0]
 receipt=json.loads((rd/'binary_validation.json').read_text());assert receipt['n_detected']==40 and receipt['n_background']==10 and receipt['n_called']==30 and receipt['cycle']==3
 assert not receipt['error_correction'] and not receipt['phasing_correction']

@pytest.mark.parametrize('change',['channels','cycles'])
def test_refuse_wrong_acquisition_or_dye_order(tmp_path,change):
 path,rd,cfg,panel=prepare(tmp_path)
 if change=='channels':cfg['channels']=['cy5','430','TxRed','FAM','cy3']
 else:cfg['cycles']=[1,2]
 path.write_text(yaml.safe_dump(cfg))
 with pytest.raises(ValueError):vis_pipeline.main(['--config',str(path),'--run-id',cfg['run_id']])
 assert not (rd/'mapped_genes.csv').exists()
