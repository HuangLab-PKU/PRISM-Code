import os 
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'exapmle_data'

def expression_matrix2D(RUN_ID):
    src_dir = BASE_DIR / f'{RUN_ID}_processed'
    read_dir = src_dir / 'readout'
    seg_dir = src_dir / 'segmented'

    # Read nucleus position
    centroids = pd.read_csv(seg_dir/'dapi_centroids_filtered.csv').to_numpy()

    # Assign RNA to its nearest nucleus
    rna_df = pd.read_csv(read_dir/'mapped_genes.csv')
    rna_pos = rna_df[['Y', 'X']].to_numpy()
    tree = KDTree(centroids)
    _, indices = tree.query(rna_pos, k=1, distance_upper_bound=100)
    rna_df['Cell Index'] = indices
    rna_df = rna_df[rna_df['Cell Index'] < centroids.shape[0]]

    # Generate expression matrix
    match_df = rna_df.copy()
    match_df['Count'] = np.ones(len(match_df))
    match_df_group = match_df[['Cell Index','Gene','Count']].groupby(['Cell Index','Gene']).count()
    matrix = match_df_group.unstack().fillna(0)
    matrix.columns = matrix.columns.droplevel()
    matrix.columns.name = None
    matrix.index.name = None

    matrix.to_csv(seg_dir / 'expression_matrix.csv')

if __name__ == '__main__':
    print(RUN_ID)
    seg_dir = BASE_DIR / f'{RUN_ID}_processed' / 'segmented'
    # copy this file to the dest_dir
    current_file_path = os.path.abspath(__file__)
    target_file_path = os.path.join(seg_dir, os.path.basename(current_file_path))
    try: shutil.copy(current_file_path, target_file_path)
    except shutil.SameFileError: print('The file already exists in the destination directory.')
    except PermissionError: print("Permission denied: Unable to copy the file.")
    except FileNotFoundError: print("File not found: Source file does not exist.")
    except Exception as e: print(f"An error occurred: {e}")
    
    expression_matrix2D(RUN_ID)