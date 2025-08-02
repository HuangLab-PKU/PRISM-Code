"""
Cell Segmentation Runner Script
Simple interface for cell segmentation using unified segmentation module
"""
import os
import sys
import shutil
from pathlib import Path

# Add src to path
package_path = r'path_to_PRISM_code_src'
if package_path not in sys.path: sys.path.append(package_path)

from cell_segmentation.unified_segmentation import UnifiedSegmentation
from cell_segmentation.segmentation_config import SegmentationConfig


def run_segmentation(base_dir: str, run_id: str, cell_image_name: str = 'cyc_1_DAPI.tif',
                    method: str = 'auto', dimension: str = 'auto'):
    """
    Run cell segmentation with specified parameters
    
    Args:
        base_dir: Base directory containing processed data
        run_id: Experiment run ID
        cell_image_name: Name of the cell image file
        method: Segmentation method ('watershed', 'stardist', 'auto')
        dimension: Image dimension ('2d', '3d', 'auto')
    """
    # Setup configuration
    config = SegmentationConfig()
    config.general_params['base_dir'] = Path(base_dir)
    config.general_params['run_id'] = run_id
    config.general_params['cell_image_name'] = cell_image_name
    
    # Initialize segmenter
    segmenter = UnifiedSegmentation(config)
    
    # Setup paths
    src_dir = config.general_params['base_dir'] / f'{run_id}_processed'
    stc_dir = src_dir / 'stitched'
    seg_dir = src_dir / 'segmented'
    
    image_path = stc_dir / cell_image_name
    
    # Check if image exists
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return False
    
    try:
        print(f"Starting segmentation...")
        print(f"Image: {image_path}")
        print(f"Method: {method}")
        print(f"Dimension: {dimension}")
        
        # Run segmentation
        results = segmenter.segment(
            image_path=image_path,
            method=method,
            dimension=dimension,
            output_dir=seg_dir
        )
        
        # Print results
        print(f"\nSegmentation completed successfully!")
        print(f"Method used: {results['method']}")
        print(f"Dimension: {results['dimension']}")
        print(f"Number of cells detected: {len(results['dataframe'])}")
        print(f"Results saved to: {seg_dir}")
        
        return True
        
    except Exception as e:
        print(f"Segmentation failed: {e}")
        return False


def main():
    """Main function with example configuration"""
    
    # Example configuration - modify these paths for your data
    BASE_DIR = 'path_to_processed_dataset'
    RUN_ID = 'example_data'
    CELL_IMAGE_NAME = 'cyc_1_DAPI.tif'
    
    # Segmentation parameters
    METHOD = 'auto'      # 'watershed', 'stardist', 'auto'
    DIMENSION = 'auto'   # '2d', '3d', 'auto'
    
    print("Cell Segmentation Script")
    print("=" * 50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Run ID: {RUN_ID}")
    print(f"Cell image: {CELL_IMAGE_NAME}")
    print("=" * 50)
    
    # Copy this script to output directory for reference
    try:
        seg_dir = Path(BASE_DIR) / f'{RUN_ID}_processed' / 'segmented'
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        current_file_path = Path(__file__).resolve()
        target_file_path = seg_dir / current_file_path.name
        
        shutil.copy(current_file_path, target_file_path)
        print(f"Script copied to: {target_file_path}")
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")
    
    # Run segmentation
    success = run_segmentation(
        base_dir=BASE_DIR,
        run_id=RUN_ID,
        cell_image_name=CELL_IMAGE_NAME,
        method=METHOD,
        dimension=DIMENSION
    )
    
    if success:
        print("\nSegmentation completed successfully!")
    else:
        print("\nSegmentation failed. Please check the error messages above.")


if __name__ == '__main__':
    main()