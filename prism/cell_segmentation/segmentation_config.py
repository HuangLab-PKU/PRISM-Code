"""
Configuration class for cell segmentation
"""
from pathlib import Path


class SegmentationConfig:
    """Configuration class for segmentation parameters"""
    
    def __init__(self):
        # Watershed parameters
        self.watershed_params = {
            'min_region': 400,
            'min_cell_size': 400,
            'block_size': 20000,
            'block_stride': None,  # Will be calculated based on block_size
            'offset_value': 255,
            'threshold_block_size': 251,
            'min_distance': 7,
            'dilate_kernel': 3,
            'erode_kernel': 2
        }
        
        # StarDist 2D parameters
        self.stardist_2d_params = {
            'model_path': Path(__file__).parent.parent / 'models',
            'model_name': '2D_versatile_fluo',
            'block_size': [5000, 5000],
            'min_overlap': [448, 448],
            'context': [94, 94],
            'min_cell_size': 400,
            'predict_kwargs': {'verbose': 0}
        }
        
        # StarDist 3D parameters
        self.stardist_3d_params = {
            'model_path': Path(__file__).parent.parent / 'models',
            'model_name': 'stardist_nucleus_3D',
            'max_size': 512 * 512 * 150,
            'min_overlap': [30, 90, 90],
            'context': [20, 40, 40],
            'predict_kwargs': {'verbose': 0}
        }
        
        # General parameters
        self.general_params = {
            'base_dir': Path('path_to_processed_dataset'),
            'run_id': 'example_data',
            'cell_image_name': 'cyc_1_DAPI.tif',
            'duplicate_distance': 5
        }