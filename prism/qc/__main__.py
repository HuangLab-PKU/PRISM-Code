"""Backfill QC from accepted tables only; never repeat detection or fitting."""
from pathlib import Path
import argparse,json,re
import pandas as pd
import yaml
from .reports import generate_readout_qc,generate_gene_calling_qc,image_metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--readout',action='store_true')
    p.add_argument('--gene-calling',action='store_true')
    a=p.parse_args(argv)
    run=a.run_dir
    rd=run/'readout'
    rid=run.name.removesuffix('_processed')
    readout=a.readout or not a.gene_calling
    calling=a.gene_calling or not a.readout
    shape,pixel=image_metadata(run/'stitched')
    if readout:
        params=yaml.safe_load((rd/'readout_params.yaml').read_text(encoding='utf-8-sig'))
        if params.get('RUN_ID',rid)!=rid:raise ValueError('Readout metadata belongs to another run')
        channels=params['CHANNELS']
        counts=None
        exact=None
        log=rd/'readout.log'
        if log.is_file():
            text=log.read_text(encoding='utf-8-sig')
            matches={ch:re.search(r'  '+re.escape(ch)+r': (\d+) spots detected',
                text) for ch in channels}
            if all(matches.values()):counts={ch:int(m[1]) for ch,m in matches.items()}
            m=re.search(r'Total coordinates after removing exact duplicates: (\d+)',text)
            if m:exact=int(m[1])
        cycles=None
        native=rd/'binary_validation.json'
        if native.exists():
            receipt=json.loads(native.read_text())
            if receipt.get('run_id')!=rid:raise ValueError('Calling metadata belongs to another run')
            if 'cycle' in receipt:cycles=[receipt['cycle']]
        report=generate_readout_qc(rd/'intensity.csv',rd/'position.csv',rd,rid,channels,
                                   img_shape=shape,pixel_size_um=pixel,
                                   detection_counts=counts,
                                       exact_unique=exact,
                                       acquired_cycles=cycles)
        if report['input_rows']!=params['n_spots']:raise ValueError('QC/readout metadata count mismatch')
    if calling:
        if (rd/'vis_binary.lock').exists():raise ValueError('Active decode source; wait for actual producer exit before QC backfill')
        candidates=[rd/'mapping_postcode.csv',rd/'mapping_gmm.csv',rd/'spots_called.csv',rd/'predictions.csv']
        source=next((f for f in candidates if f.is_file()),None)
        if source is None:raise FileNotFoundError('No accepted calling table to backfill')
        features=None
        excluded=[]
        detected=None
        diagnostics={}
        model=rd/'binary_model.json'
        if model.exists():diagnostics=json.loads(model.read_text())
        counts=rd/'gene_counts.csv'
        if counts.exists():
            frame=pd.read_csv(counts,keep_default_na=False)
            col='gene' if 'gene' in frame else 'Gene' if 'Gene' in frame else None
            if col:
                features=frame[col].tolist()
                if 'class' in frame:excluded=frame.loc[frame['class']=='unprinted',col].tolist()
        summary=rd/'calling_summary.yaml'
        if summary.exists():detected=yaml.safe_load(summary.read_text()).get('n_detected')
        elif (rd/'readout_params.yaml').exists():detected=yaml.safe_load((rd/'readout_params.yaml').read_text())['n_spots']
        generate_gene_calling_qc(source,rd,rid,diagnostics=diagnostics,features=features,
                                 excluded_features=excluded,detected_spots=detected,
                                 img_shape=shape,pixel_size_um=pixel)
    print(json.dumps({'status':'QC_BACKFILL_COMPLETE','run_id':rid,'readout':readout,
                      'gene_calling':calling,'numerical_stages_replayed':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
