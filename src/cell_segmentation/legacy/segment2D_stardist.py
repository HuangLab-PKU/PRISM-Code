import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import sum
from tifffile import imread
from tifffile import imwrite

from stardist.models import StarDist2D
from csbdeep.utils import normalize


BLOCK_SIZE = [5000, 5000]
MIN_OVERLAP = [448, 448]
CONTEXT = [94, 94]
MIN_CELL_SIZE = 400
BASE_DIR = Path('path_to_your_data')
RUN_ID = 'Exapmle_data'


def segment2D_stardist(img):
    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, info = model.predict_instances_big(normalize(img), axes='YX',
                block_size=BLOCK_SIZE, min_overlap=MIN_OVERLAP, context=CONTEXT,
                labels_out_dtype=np.uint16, show_progress=True, predict_kwargs={'verbose': 0},)    

    def measure_areas(labels):
        # The index array should start from 1 up to the maximum label because label 0 is usually the background.
        areas = sum(labels > 0, labels, index=np.arange(1, labels.max() + 1))
        return areas
    
    def filter_small_labels(labels, min_size):
        # Measure areas
        areas = measure_areas(labels)
        mask = np.in1d(labels.ravel(), np.where(areas >= min_size)[0] + 1).reshape(labels.shape)
        filtered_labels = labels * mask
        return filtered_labels

    filtered_labels = filter_small_labels(labels, MIN_CELL_SIZE)
    info_index = np.unique(filtered_labels[filtered_labels>0])-1
    centroid = info['points'][info_index]
    prob = info['prob'][info_index]
    dapi_predict = pd.DataFrame(centroid, columns=['Y','X'], index=info_index)
    dapi_predict['prob'] = prob
    return labels, dapi_predict


def segment_pipeline(run_id):
    base_dir = BASE_DIR / f'{run_id}'
    stc_dir = base_dir / 'stitched'
    seg_dir = base_dir / 'segmented'
    os.makedirs(seg_dir, exist_ok=True)

    cell_im_name = 'cyc_1_DAPI'
    img = imread(stc_dir/f'{cell_im_name}.tif')
    labels, dapi_predict = segment2D_stardist(img)
    
    dapi_predict.to_csv(seg_dir / 'dapi_predict.csv', index=False)
    dapi_predict[['Y','X']].to_csv(seg_dir / 'dapi_centroids.csv', index=False)
    print(len(dapi_predict), 'cells detected')
    imwrite(seg_dir/f'{cell_im_name}_stardist.tif', labels)


if __name__ == '__main__':
    seg_dir = BASE_DIR / f'{RUN_ID}' / 'segmented'
    seg_dir.mkdir(exist_ok=True)
    
    # copy this file to the dest_dir
    current_file_path = os.path.abspath(__file__)
    target_file_path = os.path.join(seg_dir, os.path.basename(current_file_path))
    try: shutil.copy(current_file_path, target_file_path)
    except shutil.SameFileError: print('The file already exists in the destination directory.')
    except PermissionError: print("Permission denied: Unable to copy the file.")
    except FileNotFoundError: print("File not found: Source file does not exist.")
    except Exception as e: print(f"An error occurred: {e}")
    
    segment_pipeline(RUN_ID)