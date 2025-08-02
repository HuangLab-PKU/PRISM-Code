import numpy as np
from tifffile import imread
from stardist import random_label_cmap, _draw_polygons
from stardist.models import Config2D, StarDist2D
from csbdeep.utils import normalize
import os
from tqdm import tqdm

def load_data(data_dir, split='training'):
    """
    Loads images and masks from the specified directory.
    """
    image_dir = os.path.join(data_dir, split, 'images')
    mask_dir = os.path.join(data_dir, split, 'masks')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])

    images = [imread(os.path.join(image_dir, f)) for f in tqdm(image_files, desc=f"Loading {split} images")]
    masks = [imread(os.path.join(mask_dir, f)) for f in tqdm(mask_files, desc=f"Loading {split} masks")]

    return images, masks

def prepare_data(images, masks):
    """
    Normalizes multi-channel images and prepares them for StarDist.
    Transposes images from (C, H, W) to (H, W, C) format.
    """
    # Transpose each image from (C, H, W) to (H, W, C) to match StarDist's expected input shape
    images_transposed = [np.transpose(img, (1, 2, 0)) for img in images]
    
    # Normalize along the new Y and X axes (0 and 1)
    axis_norm = (0, 1)
    
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(images_transposed, desc="Normalizing images")]
    Y = [m.astype(np.uint16) for m in tqdm(masks, desc="Preparing masks")]
    
    return X, Y 