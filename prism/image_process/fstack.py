from pathlib import Path
from lib.utils.os_utils import try_mkdir
from lib.utils.io_utils import imlist_to_df
from lib.utils.io_utils import get_tif_list
from tqdm import tqdm
import matlab.engine


def stack_cyc_chn(in_dir,out_dir,cyc,chn):
    """Stack images in selected cycle and channel

    Parameters
    ----------
    in_dir : str
        Path to the directory containing the images to be stacked
    out_dir : str
        Path to the directory where the stacked images will be saved
    cyc : int
        Image cycle number
    chn : str
        Image channel
    
    Returns
    -------
    None.

    """
    eng = matlab.engine.start_matlab()
    file_path = Path(__file__).parent
    eng.addpath(str(file_path))
    cyc_dir = Path(in_dir)/f'cyc_{cyc}'
    tif_list = get_tif_list(cyc_dir)
    meta_df = imlist_to_df(tif_list)
    meta_df = meta_df[meta_df['Channel']==chn]
    meta_df['Name'] = meta_df['Name'].apply(lambda x: str(cyc_dir/x))
    dest_dir = Path(out_dir)/f'cyc_{cyc}_{chn}'
    try_mkdir(dest_dir)
    groups = meta_df.groupby(['Tile'])['Name']
    for tile, names in tqdm(groups,desc=f'Stacking cycle {cyc} channel {chn}'):
        output_name = f'FocalStack_{tile:03}.tif'
        if (dest_dir/output_name).is_file():
            continue
        eng.fstack_write(list(names),str(dest_dir/output_name),nargout=0)
    eng.quit()


def stack_cyc(in_dir,out_dir,cyc,chns=['cy3','cy5','DAPI','FAM','TxRed']):
    """Stack images in selected cycle

    Parameters
    ----------
    in_dir : str
        Path to the directory containing the images to be stacked
    out_dir : str
        Path to the directory where the stacked images will be saved
    cyc : int
        Image cycle number
    
    Returns
    -------
    None.

    """
    for chn in chns:
        cyc_dir = Path(in_dir)/f'cyc_{cyc}'
        if list(cyc_dir.glob(f'C*-T*-{chn}-Z*.tif')):
            stack_cyc_chn(in_dir,out_dir,cyc,chn)


if __name__ == '__main__':
    pass