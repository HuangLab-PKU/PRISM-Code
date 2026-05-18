"""
Unified cell segmentation module
Supports multiple segmentation methods and automatic 2D/3D detection
"""
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from math import ceil
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from scipy.ndimage import sum as nd_sum

from skimage.io import imread, imsave
from skimage import measure
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tifffile import imread as tiff_imread, imwrite

from .segmentation_config import SegmentationConfig

try:
    from stardist.models import StarDist2D, StarDist3D
    from stardist import random_label_cmap
    from csbdeep.utils import normalize
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    print("Warning: StarDist not available. Only watershed method can be used.")


class UnifiedSegmentation:
    """Unified segmentation class"""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._stardist_2d_model = None
        self._stardist_3d_model = None
        
    def detect_image_dimension(self, image_path: Path) -> str:
        """Automatically detect image dimension"""
        img = tiff_imread(image_path)
        if img.ndim == 2:
            return '2d'
        elif img.ndim == 3:
            # Check if it's a true 3D image (z dimension > 1)
            if img.shape[0] > 1:
                return '3d'
            else:
                return '2d'
        else:
            raise ValueError(f"Unsupported image dimension: {img.ndim}")
    
    def _get_stardist_2d_model(self):
        """Get StarDist 2D model (lazy loading)"""
        if not STARDIST_AVAILABLE:
            raise ImportError("StarDist is not available")
        if self._stardist_2d_model is None:
            self._stardist_2d_model = StarDist2D.from_pretrained(
                self.config.stardist_2d_params['model_name']
            )
        return self._stardist_2d_model
    
    def _get_stardist_3d_model(self):
        """Get StarDist 3D model (lazy loading)"""
        if not STARDIST_AVAILABLE:
            raise ImportError("StarDist is not available")
        if self._stardist_3d_model is None:
            self._stardist_3d_model = StarDist3D(
                None, 
                name=self.config.stardist_3d_params['model_name'],
                basedir=self.config.stardist_3d_params['model_path']
            )
        return self._stardist_3d_model
    
    def segment_watershed_2d(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """2D segmentation using Watershed method"""
        params = self.config.watershed_params
        
        # Adaptive thresholding
        threshold_im = threshold_local(
            img, 
            params['threshold_block_size'], 
            offset=-params['offset_value']
        )
        threshold_sup = np.quantile(threshold_im, 0.98)
        threshold_im[threshold_im > threshold_sup] = threshold_sup
        
        # Create mask
        bool_mask = img > threshold_im
        bool_mask = remove_small_objects(bool_mask, params['min_region'])
        
        # Morphological operations
        cells = np.zeros(img.shape, dtype=np.uint8)
        cells[img > threshold_im] = 1000
        cells = cv2.dilate(cells, disk(params['dilate_kernel']))
        cells = cv2.erode(cells, disk(params['erode_kernel']))
        cells = cells > 0
        cells = remove_small_objects(cells, params['min_region'])
        
        # Distance transform and watershed
        distance = ndi.distance_transform_edt(cells)
        coordinates = peak_local_max(distance, min_distance=params['min_distance'])
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coordinates.T)] = True
        markers = measure.label(mask)
        segmented = watershed(-distance, markers, mask=cells)
        
        # Filter small cells
        unique, counts = np.unique(segmented, return_counts=True)
        small_labels = unique[counts < params['min_cell_size']]
        segmented[np.isin(segmented, small_labels)] = 0.0
        
        # Update coordinates
        coordinates = coordinates[segmented[coordinates[:, 0], coordinates[:, 1]] != 0.0]
        
        return coordinates, segmented
    
    def segment_stardist_2d(self, img: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """2D segmentation using StarDist"""
        model = self._get_stardist_2d_model()
        params = self.config.stardist_2d_params
        
        # Prediction
        labels, info = model.predict_instances_big(
            normalize(img), 
            axes='YX',
            block_size=params['block_size'],
            min_overlap=params['min_overlap'],
            context=params['context'],
            labels_out_dtype=np.uint16,
            show_progress=True,
            predict_kwargs=params['predict_kwargs']
        )
        
        # Filter small cells
        filtered_labels = self._filter_small_labels_stardist(labels, params['min_cell_size'])
        info_index = np.unique(filtered_labels[filtered_labels > 0]) - 1
        centroid = info['points'][info_index]
        prob = info['prob'][info_index]
        
        # Create result DataFrame
        result_df = pd.DataFrame(centroid, columns=['Y', 'X'], index=info_index)
        result_df['prob'] = prob
        
        return filtered_labels, result_df
    
    def segment_stardist_3d(self, img: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """3D segmentation using StarDist"""
        model = self._get_stardist_3d_model()
        params = self.config.stardist_3d_params
        
        # Data preprocessing
        n_channel = 1 if img.ndim == 3 else img.shape[-1]
        axis_norm = (0, 1, 2)
        img_normalized = normalize(img, 1, 99.8, axis=axis_norm)
        
        # Calculate block size
        block_size_x = int(img.shape[0] * 3 / 4)
        block_size_y = int(np.sqrt(params['max_size'] / block_size_x))
        block_size_z = int(np.sqrt(params['max_size'] / block_size_x))
        block_size = [block_size_x, block_size_y, block_size_z]
        
        # Prediction
        predict_image, poly = model.predict_instances_big(
            img_normalized,
            axes='ZYX',
            block_size=block_size,
            min_overlap=params['min_overlap'],
            context=params['context'],
            labels_out_dtype=np.uint16,
            show_progress=True,
            predict_kwargs=params['predict_kwargs']
        )
        
        # Process results
        centroids = poly['points']
        result_df = pd.DataFrame(centroids, columns=['Z', 'X', 'Y'])
        
        return predict_image, result_df
    
    def _filter_small_labels_stardist(self, labels: np.ndarray, min_size: int) -> np.ndarray:
        """Filter small labels in StarDist results"""
        areas = nd_sum(labels > 0, labels, index=np.arange(1, labels.max() + 1))
        mask = np.in1d(labels.ravel(), np.where(areas >= min_size)[0] + 1).reshape(labels.shape)
        return labels * mask
    
    def block_segment_watershed(self, img: np.ndarray, out_dir: Path, 
                               output: bool = True, output_original: bool = True):
        """Block processing for large images (watershed only)"""
        params = self.config.watershed_params
        block_size = params['block_size']
        block_stride = params['block_stride'] or (block_size - 100)
        
        overlap = block_stride - block_size
        y, x = img.shape
        y_steps = ceil((y - overlap) / block_stride)
        x_steps = ceil((x - overlap) / block_stride)
        
        print(f'Segmenting image with {img.shape}...')
        print(f'A total of {y_steps} x {x_steps} blocks...')
        
        for y_step in tqdm(range(y_steps)):
            for x_step in range(x_steps):
                block = img[y_step*block_stride:y_step*block_stride+block_size,
                           x_step*block_stride:x_step*block_stride+block_size]
                coordinates, segmented = self.segment_watershed_2d(block)
                coordinates += [y_step*block_stride, x_step*block_stride]
                
                np.savetxt(
                    out_dir / f'centroids_y_{y_step}_x_{x_step}.csv',
                    coordinates, fmt='%d', delimiter=','
                )
                
                if output:
                    imsave(
                        out_dir / f'segmented_y_{y_step}_x_{x_step}.tif',
                        segmented, check_contrast=False
                    )
                if output_original:
                    imsave(
                        out_dir / f'centroids_y_{y_step}_x_{x_step}_original.tif',
                        block, check_contrast=False
                    )
    
    def remove_duplicates(self, coordinates: np.ndarray, distance: float = None) -> np.ndarray:
        """Remove duplicate coordinate points"""
        if distance is None:
            distance = self.config.general_params['duplicate_distance']
            
        tree = KDTree(coordinates)
        pairs = tree.query_pairs(distance)
        neighbors = {}
        
        for i, j in pairs:
            if i not in neighbors:
                neighbors[i] = set([j])
            else:
                neighbors[i].add(j)
            if j not in neighbors:
                neighbors[j] = set([i])
            else:
                neighbors[j].add(i)
        
        keep = []
        discard = set()
        nodes = set([s[0] for s in pairs] + [s[1] for s in pairs])
        
        for node in nodes:
            if node not in discard:
                keep.append(node)
                discard.update(neighbors.get(node, set()))
        
        return np.delete(coordinates, list(discard), axis=0)
    
    def combine_centroids(self, in_dir: Path, out_dir: Path):
        """Combine block processing results"""
        dapi_centroids = None
        centroids_list = list(in_dir.glob('centroids_y_*_x_*.csv'))
        
        for path in centroids_list:
            centroids = np.loadtxt(path, delimiter=',', dtype=int)
            if len(centroids) == 0:
                continue
            elif len(centroids.shape) == 1:
                centroids = centroids[np.newaxis, :]
            
            if dapi_centroids is None:
                dapi_centroids = centroids
            else:
                dapi_centroids = np.unique(
                    np.concatenate((dapi_centroids, centroids), axis=0), axis=0
                )
        
        print(f'Total number of centroids: {dapi_centroids.shape[0]}')
        dapi_centroids = self.remove_duplicates(dapi_centroids)
        print(f'Number of unique centroids: {dapi_centroids.shape[0]}')
        
        np.savetxt(out_dir / 'dapi_centroids.csv', dapi_centroids, fmt='%d', delimiter=',')
    
    def segment(self, image_path: Path, method: str = 'auto', dimension: str = 'auto',
                output_dir: Path = None) -> Dict[str, Any]:
        """
        Unified segmentation interface
        
        Args:
            image_path: Path to input image
            method: Segmentation method ('watershed', 'stardist', 'auto')
            dimension: Image dimension ('2d', '3d', 'auto')
            output_dir: Output directory
            
        Returns:
            Dictionary containing segmentation results
        """
        # Auto-detect dimension
        if dimension == 'auto':
            dimension = self.detect_image_dimension(image_path)
        
        # Auto-select method
        if method == 'auto':
            method = 'stardist' if STARDIST_AVAILABLE else 'watershed'
        
        # Validate method and dimension compatibility
        if method == 'watershed' and dimension == '3d':
            raise ValueError("Watershed method only supports 2D images")
        
        if method == 'stardist' and not STARDIST_AVAILABLE:
            raise ImportError("StarDist is not available")
        
        # Read image
        img = tiff_imread(image_path)
        
        # Execute segmentation
        results = {}
        
        if method == 'watershed':
            if dimension == '2d':
                coordinates, segmented = self.segment_watershed_2d(img)
                results = {
                    'method': 'watershed',
                    'dimension': '2d',
                    'coordinates': coordinates,
                    'labels': segmented,
                    'dataframe': pd.DataFrame(coordinates, columns=['Y', 'X'])
                }
        
        elif method == 'stardist':
            if dimension == '2d':
                labels, df = self.segment_stardist_2d(img)
                results = {
                    'method': 'stardist',
                    'dimension': '2d',
                    'coordinates': df[['Y', 'X']].values,
                    'labels': labels,
                    'dataframe': df
                }
            elif dimension == '3d':
                labels, df = self.segment_stardist_3d(img)
                results = {
                    'method': 'stardist',
                    'dimension': '3d',
                    'coordinates': df[['Z', 'X', 'Y']].values,
                    'labels': labels,
                    'dataframe': df
                }
        
        # Save results
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            stem = image_path.stem
            
            # Save label image
            imwrite(output_dir / f'{stem}_{method}_{dimension}_labels.tif', results['labels'])
            
            # Save coordinates
            results['dataframe'].to_csv(output_dir / f'{stem}_{method}_{dimension}_centroids.csv', index=False)
            
            # Special handling: save compatible format
            if method == 'stardist' and dimension == '2d':
                results['dataframe'][['Y', 'X']].to_csv(
                    output_dir / 'dapi_centroids.csv', index=False
                )
                results['dataframe'].to_csv(
                    output_dir / 'dapi_predict.csv', index=False
                )
        
        return results