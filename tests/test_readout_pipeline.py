"""Exercise real tiled TIFF readout and failure propagation."""
import numpy as np
import pandas as pd
import pytest
import yaml
from tifffile import imwrite
from prism.readout import pipeline


def test_actual_tiled_readout_has_matched_ids_and_finite_intensity(tmp_path):
    st=tmp_path/'stitched';st.mkdir();rd=tmp_path/'readout';rd.mkdir()
    a=np.full((128,128),100,dtype=np.uint16)
    centers=[(20,20),(55,55),(111,111)]
    from scipy.ndimage import gaussian_filter
    signal=np.zeros((128,128),dtype=float)
    for y,x in centers:signal[y,x]=30000
    a=(a+gaussian_filter(signal,1.)).astype(np.uint16)
    for channel in ('cy3','cy5'):imwrite(st/f'cyc_1_{channel}.tif',a)
    cfg={'base_dir':str(tmp_path),'channels':[{'file':f'cyc_1_{ch}.tif','name':ch} for ch in ('cy3','cy5')],
         'detection_method':'tophat','block_size':[64,64],'block_overlap':[8,8],
         'n_workers':1,'detection_snr':{'cy3':2,'cy5':2}}
    (tmp_path/'readout.yaml').write_text(yaml.safe_dump(cfg))
    pipeline.load_config(tmp_path)
    positions,intensity=pipeline.main('FIXTURE',st,rd)
    assert np.array_equal(positions['index'],intensity['index'])
    assert np.isfinite(intensity.to_numpy()).all()
    assert len(positions)==3
    assert set(map(tuple,positions[['Y','X']].to_numpy()))==set(centers)


def test_load_config_does_not_inherit_previous_run_settings(tmp_path):
    base={'base_dir':str(tmp_path),'channels':[{'file':'cyc_1_cy3.tif','name':'G'}]}
    (tmp_path/'readout.yaml').write_text(yaml.safe_dump({**base,'detection_prob_thresh':.01,'n_workers':1}))
    pipeline.load_config(tmp_path)
    (tmp_path/'readout.yaml').write_text(yaml.safe_dump(base))
    pipeline.load_config(tmp_path)
    assert pipeline.DETECTION_PROB_THRESH is None and pipeline.N_WORKERS==4


def test_intensity_worker_does_not_hide_errors(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('fixture reader failure')
    monkeypatch.setattr(pipeline,'read_intensity_tophat',fail)
    with pytest.raises(RuntimeError,match='fixture reader failure'):
        pipeline._read_intensity_in_block(('G',np.ones((4,4)),np.array([0]),np.array([[1,1]]),3,1))



def test_detection_worker_propagates_empty_message_errors(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError()
    monkeypatch.setattr(pipeline,'get_spot_coordinates',fail)
    with pytest.raises(RuntimeError):
        pipeline._detect_spots_in_block(('G',np.ones((4,4)),0,0,'tophat',{}))
