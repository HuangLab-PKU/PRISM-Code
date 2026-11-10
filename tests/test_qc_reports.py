"""QC contracts: real figures, conservation, missing confidence and required pipeline products."""
import json
import numpy as np
import pandas as pd
import pytest
from prism.qc import generate_readout_qc,generate_gene_calling_qc,require_qc_artifacts


def readout(n=4):
    ids=np.arange(n)
    return (pd.DataFrame({'index':ids,'a':ids*10.,'b':ids*20.}),
            pd.DataFrame({'index':ids,'Y':ids*2.,'X':ids*3.}))


def test_readout_actual_png_and_all_row_count_conservation(tmp_path):
    values,positions=readout()
    r=generate_readout_qc(values,positions,tmp_path,'sample',['a','b'],img_shape=(20,20),
                          detection_counts={'a':5,'b':4},exact_unique=6,acquired_cycles=[3])
    assert r['input_rows']==r['spatial_rows']==4 and r['acquired_cycles']==[3]
    assert np.asarray(r['intensity_histogram_counts']).sum(axis=1).tolist()==[4,4]
    assert r['funnel']['per_channel_detections']=={'a':5,'b':4}
    assert require_qc_artifacts(tmp_path,'readout',expected_rows=4)['status']=='QC_COMPLETE'
    assert values.columns.tolist()==['index','a','b']


def test_chunked_and_dataframe_QC_have_same_counts_and_stable_sample(tmp_path,monkeypatch):
    import prism.qc.reports as q
    monkeypatch.setattr(q,'CHUNK',2)
    monkeypatch.setattr(q,'SAMPLE',3)
    values,positions=readout(8)
    values.to_csv(tmp_path/'i.csv',index=False)
    positions.to_csv(tmp_path/'p.csv',index=False)
    a=generate_readout_qc(values,positions,tmp_path/'memory','sample',['a','b'])
    b=generate_readout_qc(tmp_path/'i.csv',tmp_path/'p.csv',tmp_path/'csv','sample',['a','b'])
    assert a['intensity_histogram_counts']==b['intensity_histogram_counts']
    assert a['correlation_sample']==b['correlation_sample']
    assert a['log1p_correlation']==b['log1p_correlation']
    assert a['correlation_sample']['rows']==3


def test_readout_rejects_misaligned_ids_and_nonfinite_values(tmp_path):
    values,positions=readout()
    positions['index']+=1
    with pytest.raises(ValueError,match='identities'):
        generate_readout_qc(values,positions,tmp_path,'sample',['a','b'])
    positions['index']-=1
    values.loc[1,'a']=np.nan
    with pytest.raises(ValueError,match='intensity'):
        generate_readout_qc(values,positions,tmp_path,'sample',['a','b'])
    assert not (tmp_path/'readout_qc.json').exists()


def test_missing_pure_channel_confidence_is_not_fabricated(tmp_path):
    frame=pd.DataFrame({'gene':['A','B','Background','U'],
                        'confidence':[.99,np.nan,.8,.7],
                        'Y':[1,2,3,4],'X':[3,4,5,6]})
    before=frame.copy(deep=True)
    r=generate_gene_calling_qc(frame,tmp_path,'sample',features=['A','B','U','Zero'],
                               excluded_features=['U'],detected_spots=7,
                               diagnostics={'losses':[3.,2.,1.]})
    assert r['input_rows']==4 and r['assigned_target_rows']==2
    assert r['rows_without_reported_confidence']==1
    assert r['target_rows_with_reported_confidence']==1
    assert r['not_emitted_rows']==3 and not r['missing_values_imputed']
    assert r['targets_above_Q_threshold'][r['posterior_Q_thresholds'].index(20)]==1
    assert next(x for x in r['per_feature'] if x['feature']=='Zero')['count']==0
    pd.testing.assert_frame_equal(frame,before)
    require_qc_artifacts(tmp_path,'gene_calling',expected_rows=4)


def test_absent_posterior_entropy_and_coordinates_are_explicit(tmp_path):
    r=generate_gene_calling_qc(pd.DataFrame({'predicted_label':[0,1,1]}),tmp_path,'sample')
    assert r['rows_without_reported_confidence']==3
    assert r['spatial_target_rows'] is None and r['fitting_objective']==[]
    assert np.asarray(r['probability_histogram_target_and_non_target']).sum()==0
    assert require_qc_artifacts(tmp_path,'gene_calling')['status']=='QC_COMPLETE'


def test_figure_tampering_or_missing_figure_blocks_acceptance(tmp_path):
    values,positions=readout()
    generate_readout_qc(values,positions,tmp_path,'sample',['a','b'])
    (tmp_path/'readout_qc.png').write_bytes(b'not a figure')
    with pytest.raises(ValueError,match='identity'):
        require_qc_artifacts(tmp_path,'readout')
    (tmp_path/'readout_qc.png').unlink()
    with pytest.raises(FileNotFoundError):require_qc_artifacts(tmp_path,'readout')


def test_empty_calling_result_still_produces_explicit_empty_QC(tmp_path):
    r=generate_gene_calling_qc(pd.DataFrame(columns=['Gene',
        'Probability']),
        tmp_path,
        'empty',
        features=['A'])
    assert r['input_rows']==0 and r['assigned_target_rows']==0
    assert r['per_feature'][0]['count']==0
    require_qc_artifacts(tmp_path,'gene_calling',expected_rows=0)


def test_generic_classifier_entry_emits_QC_from_real_top_two_columns(tmp_path):
    import logging
    from types import SimpleNamespace
    from prism.gene_calling.pipeline import SignalClassificationPipeline
    from prism.gene_calling.base import ClassificationResult
    pipe=object.__new__(SignalClassificationPipeline)
    pipe.logger=logging.getLogger('qc-fixture')
    pipe.config={'classification':{}}
    pipe.method=SimpleNamespace(get_model_info=lambda:{})
    source=pd.DataFrame({'a':[1.,2.],'b':[3.,4.]})
    probabilities=np.array([[.99,.01],[.05,.95]])
    result=ClassificationResult(np.array([0,
        1]),
        probabilities=probabilities.copy(),
        metadata={'codebook':pd.DataFrame({'gene':['A',
        'B']})})
    pipe.load_data=lambda *args:source
    pipe.fit=lambda *args:None
    pipe.predict=lambda *args:result
    pipe.evaluate=lambda *args:{}
    pipe.run_full_pipeline('synthetic.csv',output_dir=tmp_path)
    report=require_qc_artifacts(tmp_path,'gene_calling',expected_rows=2)
    assert report['source_label_column']=='top1_gene' and report['label_counts']=={'A':1,'B':1}
    assert report['target_rows_with_reported_confidence']==2
    assert report['rows_without_reported_confidence']==0
    np.testing.assert_array_equal(result.probabilities,probabilities)


def test_backfill_command_only_reads_completed_tables(tmp_path):
    import yaml
    from prism.qc.__main__ import main
    run=tmp_path/'sample_processed'
    rd=run/'readout'
    rd.mkdir(parents=True)
    values,positions=readout()
    values.to_csv(rd/'intensity.csv',index=False)
    positions.to_csv(rd/'position.csv',index=False)
    (rd/'readout_params.yaml').write_text(yaml.safe_dump({'RUN_ID':'sample',
        'CHANNELS':['a',
        'b'],
        'n_spots':4}))
    calls=pd.DataFrame({'Gene':['A','B','Background','A'],'Probability':[.99,.8,.9,1.],
                        'Probability_2':[.01,.1,.05,0.],'Entropy':[.1,.5,.2,0.],
                        'Y':[0,2,4,6],'X':[0,3,6,9]})
    calls.to_csv(rd/'mapping_postcode.csv',index=False)
    before=(rd/'mapping_postcode.csv').read_bytes()
    assert main(['--run-dir',str(run)])==0
    require_qc_artifacts(rd,'readout',expected_rows=4)
    require_qc_artifacts(rd,'gene_calling',expected_rows=4)
    assert (rd/'mapping_postcode.csv').read_bytes()==before



def test_above_camera_range_is_reported_not_silently_dropped_or_rejected(tmp_path):
    values,positions=readout()
    values.loc[1,'a']=100000.
    before=values.copy(deep=True)
    report=generate_readout_qc(values,positions,tmp_path,'sample',['a','b'])
    assert report['above_65535_counts']==[1,0]
    assert np.asarray(report['intensity_histogram_counts']).sum(axis=1).tolist()==[4,4]
    pd.testing.assert_frame_equal(values,before)
