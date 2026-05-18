# from __future__ import print_function, unicode_literals, absolute_import, division

from stardist.models import StarDist3D
from stardist import random_label_cmap
from csbdeep.utils import normalize
from tifffile import imwrite
from tifffile import imread
import numpy as np
import pandas as pd
from pathlib import Path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.random.seed(6)

lbl_cmap = random_label_cmap()

# choose and load models
model = StarDist3D(None, name="stardist", basedir="./models")


dapi_name = 'cyc_1_dapi.tif'
BASE_DIR = Path('/PRISM_code/dataset/_example_dataset/processed')
RUN_ID = '20230227_test'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
seg_dir = src_dir / 'segmented'
os.makedirs(seg_dir, exist_ok=True)


def segment3D(model, dapi_name, stc_dir, seg_dir):
    # load image and preprocess
    raw_image = imread(stc_dir/dapi_name)
    n_channel = 1 if raw_image.ndim == 3 else raw_image.shape[-1]
    axis_norm = (0, 1, 2)  # normalize channels independently
    if n_channel > 1:
        print("Normalizing image channels %s." % (
            "jointly" if axis_norm is None or 2 in axis_norm else "independently"))
    raw_image = normalize(raw_image, 1, 99.8, axis=axis_norm)

    # prediction
    file_name = dapi_name.replace(".tif", "")
    max_size = 512 * 512 * 150
    block_size_x = int(raw_image.shape[0] * 3 / 4)
    block_size_y = int(np.sqrt(max_size / block_size_x))
    block_size_z = int(np.sqrt(max_size / block_size_x))
    block_size = [block_size_x, block_size_y, block_size_z]
    predict_image, poly = model.predict_instances_big(raw_image, axes='ZYX',
                                                      block_size=block_size, min_overlap=[30, 90, 90],
                                                      context=[20, 40, 40],
                                                      labels_out_dtype=np.uint16, show_progress=True, predict_kwargs={'verbose': 0},
                                                      )
    centroids_stardist = poly['points']

    # save prediction file
    predict_image_path = seg_dir / f'{file_name}_predict.tif'
    imwrite(predict_image_path, predict_image)

    # save centroid of stardist and gray scale as csv and save image
    predict_centroid_path = seg_dir / f'{file_name}_centroid.csv'
    centroids_stardist = pd.DataFrame(centroids_stardist, columns=['z_in_pix', 'x_in_pix', 'y_in_pix'])
    centroids_stardist.to_csv(predict_centroid_path)


if __name__ == '__main__':
    segment3D(model=model, dapi_name=dapi_name, stc_dir=stc_dir, seg_dir=seg_dir)
