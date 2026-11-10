"""Contracts for five combinatorial bits in one acquisition, not two rounds."""
import itertools
import numpy as np
import pytest
from prism.vis_binary import decode

def fixture():
 codes=np.asarray(list(itertools.product([0,1],repeat=5))[1:],dtype=np.float32)[:,:,None]
 rng=np.random.default_rng(4);truth=np.repeat(np.arange(32),120)
 bits=np.vstack([codes[:,:,0],np.zeros((1,5))])[truth]
 values=rng.lognormal(np.log(40+bits*3500),.10).astype(np.float32)[:,:,None]
 return codes,truth,values

def test_single_round_probabilistic_decode():
 codes,truth,values=fixture()
 result,diagnostics=decode(values,codes,[f'Target{i}' for i in range(31)],[f'{i:05b}' for i in range(1,32)],num_iter=120,batch_size=5000,inference_chunk_size=2000,verbose=False)
 calls=result.Gene.to_numpy();expected=np.asarray([f'Target{i}' for i in range(31)]+['Background'])[truth]
 assert (calls==expected).mean()>.95
 assert (calls[truth==31]=='Background').mean()>.95
 assert np.isfinite(result.Probability).all() and result.Probability.between(0,1).all()
 assert diagnostics['n_total_spots']==len(values)
 assert result.Sequence.str.fullmatch('[01]{5}|Background').all()

def test_reject_second_cycle_and_zero_gene_word():
 codes,truth,values=fixture()
 with pytest.raises(ValueError,match='N x5 x1'):
  decode(np.repeat(values,2,axis=2),codes,['x']*31,['x']*31)
 bad=codes.copy();bad[0]=0
 with pytest.raises(ValueError,match='all-dark'):
  decode(values,bad,[str(i) for i in range(31)],[str(i) for i in range(31)])

def test_reject_missing_channel_or_nonfinite_signal():
 codes,truth,values=fixture()
 with pytest.raises(ValueError,match='N x5 x1'):decode(values[:,:4],codes,['x']*31,['x']*31)
 values[0,0,0]=np.nan
 with pytest.raises(ValueError,match='invalid'):decode(values,codes,['x']*31,['x']*31)
