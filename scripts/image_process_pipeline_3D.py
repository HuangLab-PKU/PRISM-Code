import os
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from skimage.io import imread

# Set pandas options
pd.set_option('mode.chained_assignment', 'raise')

# Add PRISM code path
package_path = r'path_to_PRISM_code_src'
if package_path not in sys.path:
    sys.path.append(package_path)

from image_process.utils.io_utils import get_tif_list
from image_process.utils.image_transforms import create_ellipsoid_kernel, apply_tophat_filter
from image_process.batch_operations import resize_batch, create_3d_stacks, process_spot_detection_batch
from image_process.fstack import stack_cyc
from image_process.cidre import cidre_walk
from image_process.register import register_meta, register_manual
from image_process.stitch import patch_tiles, template_stitch, stitch_offset, stitch_manual
from image_process.coordinate_processing import shift_correction, stitch_3d, filter_coordinates_by_bounds, extract_intensity_at_coordinates
from lib.stitch import read_meta
from lib.AIRLOCALIZE.airlocalize import airlocalize


SRC_DIR = Path('path_to_raw_dataset')
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f'{RUN_ID}_processed'

# 2D workflow
aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
rsz_dir = dest_dir / 'resized'
read_dir = dest_dir / 'readout'

# 3D workflow
cid_dir = dest_dir / 'cidre'
air_dir = dest_dir / 'airlocalize_stack'

def process_2d():
    """Execute 2D image processing workflow"""
    cidre_walk(str(aif_dir), str(sdc_dir))

    rgs_dir.mkdir(exist_ok=True)
    ref_cyc, ref_chn, ref_chn_1 = 1, 'cy3', 'cy5'
    ref_dir = sdc_dir / f'cyc_{ref_cyc}_{ref_chn}'
    im_names = get_tif_list(ref_dir)

    meta_df = register_meta(str(sdc_dir), str(rgs_dir), ['cy3', 'cy5', 'DAPI'], im_names, ref_cyc, ref_chn)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    register_manual(rgs_dir/'cyc_10_DAPI', sdc_dir/'cyc_11_DAPI', rgs_dir/'cyc_11_DAPI')
    
    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn}', 6 * 7)
    stc_dir.mkdir(exist_ok=True)
    template_stitch(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_1}', stc_dir, 6, 7)

# Global parameters
extract_points_cycle = ['C001', 'C002', 'C003', 'C004']
CHANNELS = ['cy3', 'cy5']
TOPHAT_STRUCTURE = create_ellipsoid_kernel(2, 3, 3)


def process_3d():
    """Execute 3D image processing workflow"""
    # Generate corrected 3D image for each stack
    cidre_correct(str(src_dir), str(cid_dir))  # Call the appropriate CIDRE function as needed

    # Create 3D stacks and spot detection
    stack_name, file_groups = create_3d_stacks(src_dir, air_dir, CHANNELS, extract_points_cycle)
    
    # Batch process spot detection
    process_spot_detection_batch(stack_name, air_dir, extract_points_cycle)

    # Multi-channel reading
    shift_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)
    
    for tile in tqdm(stack_name.keys(), desc='Reading spots', position=0, leave=True):
        combined_candidates = pd.read_csv(air_dir / tile / 'combined_candidates.csv')
        intensity_read = combined_candidates[['z_in_pix', 'y_in_pix', 'x_in_pix']].round().astype(np.uint16).drop_duplicates()
        intensity_read = intensity_read.reset_index()
        
        for cycle in tqdm(stack_name[tile], desc=f'Processing cycles for tile {tile}', position=1, leave=False):
            with tifffile.TiffFile(air_dir / tile / cycle / 'cy3.tif') as tif:
                shape = tif.series[0].shape

            coordinates = intensity_read[['z_in_pix', 'y_in_pix', 'x_in_pix']].copy()
            coordinates = shift_correction(coordinates, shift_df, tile=int(tile[1:]), cyc=int(cycle[1:]), ref_cyc=1)
            coordinates = filter_coordinates_by_bounds(coordinates, shape)
            
            for channel in CHANNELS:
                image = apply_tophat_filter(imread(air_dir / tile / cycle / f'{channel}.tif'), TOPHAT_STRUCTURE)
                intensities = extract_intensity_at_coordinates(image, coordinates)
                coordinates[f'{cycle}_{channel}'] = intensities
                intensity_read[f'cyc_{int(cycle[1:])}_{channel}'] = coordinates[f'{cycle}_{channel}']
                
        intensity_read = intensity_read.dropna()
        intensity_read.to_csv(air_dir / tile / 'intensity_local.csv', index=None)
    

    # Stitch intensity data
    meta_df = read_meta(stc_dir)
    pattern = r'\((\d+)\, *(\d+)\)'
    meta_df['match'] = meta_df['position'].apply(lambda x: re.match(pattern, x))
    meta_df['y'] = meta_df['match'].apply(lambda x: int(x.group(2)))
    meta_df['x'] = meta_df['match'].apply(lambda x: int(x.group(1)))
    meta_df.set_index('file', inplace=True)
    
    intensity_list = []
    for tile in tqdm(stack_name.keys(), desc='Stitching'):
        signal_df = pd.read_csv(air_dir / tile / 'intensity_local.csv')
        stitched_df = stitch_3d(signal_df, meta_df, tile=int(tile[1:]))
        intensity_list.append(stitched_df)
    
    intensity = pd.concat(intensity_list, ignore_index=True)
    intensity.to_csv(read_dir / 'intensity_all.csv', index=None)


def main():
    """Main workflow control"""
    process_3d()  # Execute 3D processing


if __name__ == '__main__':
    main()