"""Configured single-acquisition VIS31 decode, after shared five-channel readout."""
from pathlib import Path
import argparse,hashlib,json,os,time,traceback
import numpy as np
import pandas as pd
import yaml
from .vis_binary import decode

def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,default=str),encoding='utf8');t.replace(p)
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--run-id',required=True);args=ap.parse_args(argv)
 cfg=yaml.safe_load(args.config.read_text(encoding='utf-8-sig'));rid=cfg['run_id']
 cycles=cfg['cycles']
 if args.run_id!=rid or len(cycles)!=1 or type(cycles[0]) is not int or cycles[0]<1 or cfg['profile']!='vis31_5bit_single_acquisition_v1':raise ValueError('Run/acquisition/profile mismatch')
 channels=cfg['channels']
 if channels!=['cy3','430','TxRed','FAM','cy5']:raise ValueError('Expected P1-P5 physical dye order; DAPI is not a decoding bit')
 panel=pd.read_csv(cfg['panel'],dtype=str,keep_default_na=False)
 if set(panel.columns)!={'gene','code_id','bits'} or len(panel)!=31 or not panel.gene.is_unique or set(panel.bits)!={f'{i:05b}' for i in range(1,32)}:raise ValueError('Expected exactly31 distinct nonzero five-bit gene assignments')
 rd=Path(cfg['base_dir'])/(rid+'_processed')/'readout';rd.mkdir(parents=True,exist_ok=True)
 outputs=['mapping_postcode.csv','mapped_genes.csv','gene_counts.csv','binary_model.json','binary_validation.json']
 if any((rd/n).exists() for n in outputs):raise ValueError('Existing decode outputs require inspected checkpoint recovery')
 lock=rd/'vis_binary.lock'
 with lock.open('x',encoding='utf8') as f:json.dump({'pid':os.getpid(),'run_id':rid},f)
 try:
  position=pd.read_csv(rd/'position.csv');intensity=pd.read_csv(rd/'intensity.csv')
  if not position['index'].is_unique or not np.array_equal(position['index'],intensity['index']):raise ValueError('Position/intensity identity mismatch')
  if not np.isfinite(position[['Y','X']].to_numpy()).all():raise ValueError('Invalid coordinates')
  bits=np.asarray([[int(x) for x in word] for word in panel.bits],dtype=np.float32)[:,:,None]
  values=intensity[channels].to_numpy(dtype=np.float32)[:,:,None]
  result,model=decode(values,bits,panel.gene.tolist(),panel.bits.tolist(),num_iter=int(cfg.get('num_iter',100)),batch_size=int(cfg.get('batch_size',100000)),inference_chunk_size=int(cfg.get('inference_chunk_size',100000)),background_prior=cfg.get('background_prior'))
  result['index']=position['index'].to_numpy();result['Y']=position.Y.to_numpy();result['X']=position.X.to_numpy()
  if len(result)!=len(position) or not np.isfinite(result[['Probability','Probability_2','Entropy']].to_numpy()).all():raise ValueError('Invalid decoder handoff')
  result.to_csv(rd/'mapping_postcode.csv',index=False);result[['index','Y','X','Gene','Probability']].to_csv(rd/'mapped_genes.csv',index=False)
  counts=result.Gene.value_counts();summary=panel.copy();summary['n_spots']=[int(counts.get(g,0)) for g in panel.gene];summary.to_csv(rd/'gene_counts.csv',index=False)
  save(rd/'binary_model.json',model)
  receipt={'status':'ROUTINE_OUTPUT_CHECKS_PASSED','run_id':rid,'profile':cfg['profile'],'cycle':cycles[0],'informative_bits':5,'channels':channels,'panel_size':31,'n_detected':len(result),'n_background':int(counts.get('Background',0)),'n_called':int(summary.n_spots.sum()),'minimum_hamming':1,'error_correction':False,'posterior_is_calibrated_error_rate':False,'phasing_correction':False,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'panel_sha256':hashlib.sha256(Path(cfg['panel']).read_bytes()).hexdigest(),'at':time.time()}
  if receipt['n_background']+receipt['n_called']!=len(result):raise ValueError('Count conservation failed')
  save(rd/'binary_validation.json',receipt);print(json.dumps(receipt),flush=True)
 except BaseException:
  save(rd/'vis_binary_failure.json',{'error':traceback.format_exc(),'at':time.time()});raise
 finally:
  if json.loads(lock.read_text())['pid']!=os.getpid():raise RuntimeError('Binary lock ownership changed')
  lock.unlink()
 return 0
if __name__=='__main__':raise SystemExit(main())

