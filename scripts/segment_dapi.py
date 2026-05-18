"""
Cell Segmentation Runner Script
Simple interface for cell segmentation using unified segmentation module
"""
import argparse
import shutil
from pathlib import Path

# Import from installed PRISM package
# After running: pip install -e . (from code/ directory)
from prism.cell_segmentation.unified_segmentation import UnifiedSegmentation
from prism.cell_segmentation.segmentation_config import SegmentationConfig


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
    

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cell segmentation on DAPI image for a given run.'
    )
    parser.add_argument(
        'run_id',
        type=str,
        help='Experiment run ID (e.g. directory name without _processed)',
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=r'\\10.10.10.1\NAS Processed Images',
        help='Base directory containing processed data (default: path_to_processed_dataset)',
    )
    parser.add_argument(
        '--nucleus-image',
        type=str,
        default='cyc_1_DAPI.tif',
        help='Name of the nucleus image file (default: cyc_1_DAPI.tif)',
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['watershed', 'stardist', 'auto'],
        default='auto',
        help='Segmentation method (default: auto)',
    )
    parser.add_argument(
        '--dimension',
        type=str,
        choices=['2d', '3d', 'auto'],
        default='auto',
        help='Image dimension (default: auto)',
    )
    return parser.parse_args()


def main():
    """Main function: parse args and run segmentation."""
    args = parse_args()

    print("Cell Segmentation Script")
    print("=" * 50)
    print(f"Base directory: {args.base_dir}")
    print(f"Run ID: {args.run_id}")
    print(f"Nucleus image: {args.nucleus_image}")
    print("=" * 50)

    # Copy this script to output directory for reference
    try:
        seg_dir = Path(args.base_dir) / f'{args.run_id}_processed' / 'segmented'
        seg_dir.mkdir(parents=True, exist_ok=True)

        current_file_path = Path(__file__).resolve()
        target_file_path = seg_dir / current_file_path.name

        shutil.copy(current_file_path, target_file_path)
        print(f"Script copied to: {target_file_path}")
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")

    # Run segmentation
    success = run_segmentation(
        base_dir=args.base_dir,
        run_id=args.run_id,
        cell_image_name=args.nucleus_image,
        method=args.method,
        dimension=args.dimension,
    )

    if success:
        print("\nSegmentation completed successfully!")
    else:
        print("\nSegmentation failed. Please check the error messages above.")


if __name__ == '__main__':
    main()