import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree
from skimage.io import imread
from datetime import datetime
from skimage.io import imsave
from draw_spots import crop_df
from cell_segmentation_adaptive import segment_cell_adaptive

def remove_duplicates(coordinates):
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(2)
    #print(f'{len(pairs)} duplicates pairs')
    neighbors = {} #dictionary of neighbors
    for i,j in pairs: #iterate over all pairs
        if i not in neighbors:
            neighbors[i] = set([j])
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = set([i])
        else:
            neighbors[j].add(i)
    #print(f'{len(neighbors)} neighbor entries')
    keep = []
    discard = set() # a list would work, but I use a set for fast member testing with `in`
    nodes = set([s[0] for s in pairs]+[s[1] for s in pairs])
    for node in nodes:
        if node not in discard: # if node already in discard set: skip
            keep.append(node) # add node to keep list
            discard.update(neighbors.get(node,set())) #add node's neighbors to discard set
    #print(f'{len(discard)} nodes discarded, {len(keep)} nodes kept')
    centroids_simplified = np.delete(coordinates, list(discard), axis=0)
    #print(f'{centroids_simplified.shape[0]} centroids after simplification')
    return centroids_simplified

def duplicate_test():
    df = pd.read_csv('/mnt/data/local_processed_data/20210929_seq1_normal_mouseBrain_processed/readout/mapped_unstack_1020.csv')
    count = 0
    for gene in tqdm(set(df['Gene'])):
        df_gene = df[df['Gene'] == gene]
        coordinates = df_gene[['Y','X']].values
        count += remove_duplicates(coordinates).shape[0]
    print(count)

def reduce_test():
    df = pd.read_csv('/mnt/data/local_processed_data/20221102_PRISM_BRAIN_FISH_2_processed/readout/mapped_genes.csv')
    df_reduced = pd.DataFrame()
    for gene in tqdm(set(df['Gene'])):
        df_gene = df[df['Gene'] == gene]
        coordinates = df_gene[['Y','X']].values
        coordinates = remove_duplicates(coordinates)
        df_gene_reduced = pd.DataFrame(coordinates, columns=['Y','X'])
        df_gene_reduced['Gene'] = gene
        df_reduced = df_reduced.append(df_gene_reduced)
    df_reduced.to_csv('/mnt/data/local_processed_data/20221102_PRISM_BRAIN_FISH_2_processed/readout/mapped_genes_reduced.csv', index=False)
    print(f'{df_reduced.shape[0]} rows')
        
    

if __name__ == '__main__':
    reduce_test()
    #for item in os.listdir('/mnt/data/local_processed_data/20210929_seq1_normal_mouseBrain_processed/readout'):
    #    print(item,datetime.fromtimestamp(os.path.getctime(os.path.join('/mnt/data/local_processed_data/20210929_seq1_normal_mouseBrain_processed/readout',item))).strftime('%Y-%m-%d %H:%M:%S'))
    #main()