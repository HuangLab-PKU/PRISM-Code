import re
import os
from tqdm import tqdm
import matlab.engine
from pathlib import Path


def cidre_correct(in_dir,out_dir,fmt='tif'):
    """Corrects the images in the input directory using CIDRE.

    Parameters
    ----------
    in_dir : str
        Path to the input directory.
    out_dir : str
        Path to the output directory.
    fmt : str
        Format of the images.

    Returns
    -------
    None.

    """
    eng = matlab.engine.start_matlab()
    cidre_path = (Path(__file__).parent)/'CIDRE'
    eng.addpath(str(cidre_path))
    eng.cidre_silent(str(Path(in_dir)/f'*.{fmt}'),out_dir,nargout=0)
    eng.quit()
    os.remove(Path(out_dir)/'cidre_model.mat')


def cidre_walk(in_dir,out_dir,fmt='tif'):
    """Corrects all the subdirectories given input directory.

    Parameters
    ----------
    in_dir : str
        Path to the input directory.
    out_dir : str
        Path to the output directory.
    fmt : str
        Format of the images.

    Returns
    -------
    None.

    """
    p = Path(in_dir).glob('cyc_[0-9]*_*')
    pattern = r'^cyc_\d+_\w+'
    sub_dirs = [x for x in p if x.is_dir()]
    sub_dirs = [x for x in sub_dirs if re.match(pattern,x.name)]
    eng = matlab.engine.start_matlab()
    cidre_path = (Path(__file__).parent)/'CIDRE'
    eng.addpath(str(cidre_path))
    for sub_dir in tqdm(sub_dirs,desc='CIDRE correcting'):
        out_sub_dir = Path(out_dir)/sub_dir.name
        src_cnt = len(list(sub_dir.glob(f'*.{fmt}')))
        dest_cnt = len(list(out_sub_dir.glob(f'*.{fmt}')))
        if src_cnt == dest_cnt:
            continue
        eng.cidre_silent(str(Path(sub_dir)/f'*.{fmt}'),str(out_sub_dir),nargout=0)
        if (out_sub_dir/'cidre_model.mat').is_file():
            os.remove(Path(out_dir)/sub_dir.name/'cidre_model.mat')
    eng.quit()
    

def main():
    pass


if __name__ == '__main__':
    main()