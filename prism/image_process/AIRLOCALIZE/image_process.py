from tabnanny import check
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from tifffile import imwrite
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
from skimage.morphology import dilation


def scale_tiff(image, scale_lower_percentile=0, scale_upper_percentile=99.99, scaleMaxRatio=0.5, verbose=True, dtype=np.uint16):
    """
    Normalize its intensity values to the range dtype / 2.
    """
    lowerbound = np.percentile(image, scale_lower_percentile)
    upperbound = np.percentile(image, scale_upper_percentile)
    normalized_image = (image - lowerbound) / (upperbound - lowerbound)
    normalized_image = np.clip(normalized_image, 0, 1)
    scaled_image = (normalized_image * np.iinfo(dtype).max * scaleMaxRatio).astype(dtype)
    if verbose: print(f"Image scaled from {lowerbound} to {upperbound}")
    return scaled_image


def substract_with_scale(image1, image2, rawtype=np.uint16, uppertype=np.int32):
    image1 = image1.astype(uppertype)
    image2 = image2.astype(uppertype)
    sub = image1 - image2
    min_val = sub.min()
    max_val = sub.max()
    scaled_subtracted_image = (sub - min_val) / (max_val - min_val) * np.iinfo(rawtype).max / 2
    scaled_subtracted_image = scaled_subtracted_image.astype(rawtype)
    return scaled_subtracted_image


def perform_DoG(image, dog_sigma=(0.5,1), enhance=True):
    """
    Apply Difference of Gaussians (DoG) to an image.
    """
    sigma1, sigma2 = dog_sigma
    image1 = gaussian_filter(image, sigma=sigma1)
    image2 = gaussian_filter(image, sigma=sigma2)
    dog = substract_with_scale(image1, image2)
    if enhance: dog = substract_with_scale(1.5 * dog, np.mean(dog))
    return dog


def build_maxima (coordinates, shape):
    Maxima = np.zeros(shape,dtype=np.uint16)
    Maxima[coordinates['z_in_pix'],coordinates['y_in_pix'],coordinates['x_in_pix']] = coordinates['integratedIntensity']
    return Maxima


def create_spherical_structure(radius):
    """
    Creates a 3D spherical structuring element with the given radius.
    
    Parameters:
    - radius: The radius of the sphere.
    
    Returns:
    - A 3D numpy array with shape (2*radius+1, 2*radius+1, 2*radius+1),
      where elements within the specified radius are True, and others are False.
    """
    # The diameter and size of the structure array
    diameter = radius * 2 + 1
    struct_size = (diameter, diameter, diameter)
    
    # Create an array of distances from the center
    arr = np.zeros(struct_size)
    center = np.array(struct_size) // 2
    for z in range(diameter):
        for y in range(diameter):
            for x in range(diameter):
                distance = np.sqrt((center[0] - z) ** 2 + (center[1] - y) ** 2 + (center[2] - x) ** 2)
                if distance <= radius:
                    arr[z, y, x] = True
                else:
                    arr[z, y, x] = False
    
    return arr.astype(np.bool_)


def save_points(points, shape, output_tiff_path, radius=3, verbose=True, dtype=np.uint16):
    points['x_in_pix'] = points['x_in_pix'].clip(lower=0).round().astype(np.uint16)
    points['y_in_pix'] = points['y_in_pix'].clip(lower=0).round().astype(np.uint16)
    points['z_in_pix'] = points['z_in_pix'].clip(lower=0).round().astype(np.uint16)
    points = points[points['integratedIntensity'] > 0]
    points.loc[:, 'integratedIntensity'] = (points['integratedIntensity'] / points['integratedIntensity'].max() * np.iinfo(dtype).max * 0.25).round().astype(dtype)

    Maxima = build_maxima(points, shape)
    # kernel = np.ones((5,5,5), dtype = np.uint16)
    kernel = create_spherical_structure(radius=radius)
    Maxima = dilation(Maxima, kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imwrite(output_tiff_path, Maxima, )


def save_points_old(df, shape, output_tiff_path, radius=3, verbose=True):
    """
    Save the points in the DataFrame to a TIFF file with dilation.
    """
    # Assuming you have the `shape` of your image from the TIFF metadata
    label_image = np.zeros(shape, dtype=np.uint16)

    # Create the binary image and dilate
    # structure = np.ones((3, 3, 3), dtype=np.bool_)
    structure = create_spherical_structure(radius=radius)

    # Vectorized setting of points in the label_image array
    # Note: Ensure that z, x, y indices are within the bounds of the image shape
    valid_points = (df['z_in_pix'] < shape[0]) & (df['x_in_pix'] < shape[1]) & (df['y_in_pix'] < shape[2])
    coords = df.loc[valid_points, ['z_in_pix', 'x_in_pix', 'y_in_pix']].astype(int).values.T
    labels = df.loc[valid_points, 'Label'].values

    # Set the labels at the coordinates
    label_image[coords[0], coords[1], coords[2]] = labels

    # Apply binary dilation to label_image
    # For each unique label, dilate separately to prevent merging
    for label in tqdm(np.unique(labels), desc='saving points', disable=not verbose):
        binary_mask = label_image == label
        dilated_mask = binary_dilation(binary_mask, structure)
        label_image[dilated_mask] = label

    # Save the dilated label image to a TIFF file
    imwrite(output_tiff_path, label_image) 
