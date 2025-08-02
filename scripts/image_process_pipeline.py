import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from skimage.util import img_as_uint
from skimage.io import imread, imsave
from scipy.io import loadmat

# 添加PRISM代码路径
package_path = r'path_to_PRISM_code_src'
if package_path not in sys.path: sys.path.append(package_path)

from image_process.utils.io_utils import get_tif_list
from image_process.os_snippets import try_mkdir
from image_process.fstack import stack_cyc
from image_process.cidre import cidre_walk
from image_process.register import register_meta, register_manual
from image_process.stitch import patch_tiles, template_stitch, stitch_offset, stitch_manual


SRC_DIR = Path('path_to_raw_dataset')
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f"{RUN_ID}_processed"
aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
rsz_dir = dest_dir / 'resized'
matdir = dest_dir / 'TileInfo.mat'


def resize_pad(img, size):
    img_resized = resize(img, size, anti_aliasing=True)
    img_padded = np.zeros(img.shape)
    y_start, x_start = (img.shape[0] - size[0]) // 2, (img.shape[1] - size[1]) // 2
    img_padded[y_start:y_start+size[0], x_start:x_start+size[1]] = img_resized
    img_padded = img_as_uint(img_padded)
    return img_padded


def resize_dir(in_dir, out_dir, chn):
    Path(out_dir).mkdir(exist_ok=True)
    chn_sizes = {'cy3': 2302, 'TxRed': 2303, 'FAM': 2301, 'DAPI': 2300}
    size = chn_sizes[chn]
    im_list = list(Path(in_dir).glob(f'*.tif'))
    for im_path in tqdm(im_list, desc=Path(in_dir).name):
        im = imread(im_path)
        im = resize_pad(im, (size, size))
        imsave(Path(out_dir)/im_path.name, im, check_contrast=False)


def resize_batch(in_dir, out_dir):
    try_mkdir(out_dir)
    cyc_paths = list(Path(in_dir).glob('cyc_*_*'))
    for cyc_path in cyc_paths:
        chn = cyc_path.name.split('_')[-1]
        if chn == 'cy5': shutil.copytree(cyc_path, Path(out_dir)/cyc_path.name, dirs_exist_ok=True)
        else: resize_dir(cyc_path, Path(out_dir)/cyc_path.name, chn)


def main():
    raw_cyc_list = list(src_dir.glob('cyc_*'))
    for cyc in raw_cyc_list:
      cyc_num = int(cyc.name.split('_')[1])
      stack_cyc(src_dir, aif_dir, cyc_num)

    # copy TileInfo.mat
    try: shutil.copy(os.path.join(src_dir, 'TileInfo.mat'), os.path.join(dest_dir, 'TileInfo.mat'))
    except Exception as e: print(f"An error occurred: {e}")

    # load Tile size
    tile_data = loadmat(matdir)
    variable_names = [name for name in tile_data.keys() if not name.startswith('__')]
    TileX, TileY = int(tile_data[variable_names[0]][0][0]), int(tile_data[variable_names[1]][0][0])

    # background correction
    cidre_walk(aif_dir, sdc_dir)

    # color correction
    resize_batch(aif_dir, rsz_dir)

    # register
    ref_cyc = 1
    ref_chn_rgs = 'FAM' # cy3, cy5
    ref_dir = rsz_dir / f'cyc_{ref_cyc}_{ref_chn_rgs}'
    im_names = get_tif_list(ref_dir)
    meta_df = register_meta(str(rsz_dir), str(rgs_dir), ['FAM', 'TxRed', 'DAPI'], im_names, ref_cyc, ref_chn_rgs)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    register_manual(rgs_dir/f'cyc_1_{ref_chn_rgs}', rsz_dir/'cyc_1_cy3', rgs_dir/'cyc_1_cy3')
    register_manual(rgs_dir/f'cyc_1_{ref_chn_rgs}', rsz_dir/'cyc_1_cy5', rgs_dir/'cyc_1_cy5')

    # stitch
    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_rgs}', TileX * TileY)
    ref_chn_stc = 'FAM' # cy5, FAM, TeRed
    stc_dir.mkdir(exist_ok=True)
    template_stitch(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_stc}', stc_dir, TileX, TileY)
    offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)
    stitch_offset(rgs_dir, stc_dir, offset_df)


if __name__ == "__main__":
    # copy this file to the dest_dir
    dest_dir.mkdir(exist_ok=True)
    current_file_path = os.path.abspath(__file__)
    target_file_path = os.path.join(dest_dir, os.path.basename(current_file_path))
    try: shutil.copy(current_file_path, target_file_path)
    except shutil.SameFileError: print('The file already exists in the destination directory.')
    except PermissionError: print("Permission denied: Unable to copy the file.")
    except FileNotFoundError: print("File not found: Source file does not exist.")
    except Exception as e: print(f"An error occurred: {e}")
    print('RUN_ID:', RUN_ID)
    main()
    shutil.rmtree(rsz_dir)
    shutil.rmtree(sdc_dir)