"""A local model directory must reach fresh Windows spawn detector workers."""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pytest,yaml
from prism.readout import pipeline

def spawn_detector(kwargs):
    from spotiflow.model import Spotiflow
    from prism.readout.pipeline import _detect_spots_in_block
    def forbidden(*args,**kwargs):raise AssertionError('from_pretrained must never run')
    Spotiflow.from_pretrained=forbidden
    result=_detect_spots_in_block(('G',np.zeros((128,128),dtype=np.float32),0,0,'spotiflow',kwargs))
    return os.getpid(),len(result[1])

def test_local_model_path_reaches_real_spawn_worker(tmp_path):
    folder=os.environ.get('TEST_LOCAL_SPOTIFLOW')
    if not folder:pytest.skip('Real offline model fixture not specified')
    cfg={'base_dir':str(tmp_path),'channels':[{'file':'cyc_1_cy3.tif','name':'G'}],'detection_method':'spotiflow','detection_prob_thresh':.01,'device':'cuda','detection_model_path':folder}
    (tmp_path/'readout.yaml').write_text(yaml.safe_dump(cfg));pipeline.load_config(tmp_path)
    kwargs=pipeline._spotiflow_detection_kwargs();assert kwargs['model_path']==str(Path(folder).resolve())
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        pid,n=pool.submit(spawn_detector,kwargs).result(timeout=120)
    assert pid!=os.getpid() and n==0
    cfg.pop('detection_model_path');(tmp_path/'readout.yaml').write_text(yaml.safe_dump(cfg));pipeline.load_config(tmp_path)
    assert pipeline.DETECTION_MODEL_PATH is None and 'model_path' not in pipeline._spotiflow_detection_kwargs()
