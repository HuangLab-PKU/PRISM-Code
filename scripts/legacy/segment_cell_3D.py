import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree


CHANNELS = ['cy5', 'TxRed', 'cy3', 'FAM']
BASE_DIR = Path('path_to_processed_dataset')
RUN_ID = 'example_data'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
seg_dir = src_dir / 'segmented'
os.makedirs(seg_dir, exist_ok=True)


# fun remove duplicates
def remove_duplicates(coordinates):
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(2)
    neighbors = {}  # dictionary of neighbors

    for i, j in pairs:  # iterate over all pairs
        if i not in neighbors:
            neighbors[i] = set([j])
        else:
            neighbors[i].add(j)

        if j not in neighbors:
            neighbors[j] = set([i])
        else:
            neighbors[j].add(i)

    keep = []
    discard = set()  # a list would work, but I use a set for fast member testing with `in`
    nodes = set([s[0] for s in pairs]+[s[1] for s in pairs])
    for node in nodes:
        if node not in discard:  # if node already in discard set: skip
            keep.append(node)  # add node to keep list
            # add node's neighbors to discard set
            discard.update(neighbors.get(node, set()))
    centroids_simplified = np.delete(coordinates, list(discard), axis=0)
    return centroids_simplified


def load_rnas(centroids, input_dir=read_dir/'mapped_genes.csv', preprocess=True):
    # load rnas
    rna_raw = pd.read_csv(input_dir)
    if not preprocess:
        rna_df = rna_raw.copy()
        rna_df = rna_df.loc[:, ~rna_df.columns.str.contains('^Unnamed')]
        print(rna_df)

    else:
        rna_raw = rna_raw[['x_in_pix', 'y_in_pix', 'z_in_pix', 'Gene']]
        print(f'ori_rna_num:\t{len(rna_raw)}')

        df = rna_raw[['x_in_pix', 'y_in_pix', 'z_in_pix', 'Gene']]
        df_reduced = pd.DataFrame()
        for gene in tqdm(set(df['Gene']), desc='deduplicating'):
            df_gene = df[df['Gene'] == gene]
            coordinates = df_gene[['x_in_pix', 'y_in_pix', 'z_in_pix']].values
            coordinates = remove_duplicates(coordinates)
            df_gene_reduced = pd.DataFrame(
                coordinates, columns=['x_in_pix', 'y_in_pix', 'z_in_pix'])
            df_gene_reduced['Gene'] = gene
            df_reduced = pd.concat([df_reduced, df_gene_reduced], axis=0)
        print(f'dedu_rna_num:\t{df_reduced.shape[0]}')

        # assign rna to nearest centroid
        rna_df = df_reduced.copy()
        rna_df['z_in_pix'] *= 3.36  # scale z to xy
        rna_pos = rna_df[['z_in_pix', 'x_in_pix', 'y_in_pix']].to_numpy()
        tree = KDTree(centroids)
        _, indices = tree.query(rna_pos, k=1, distance_upper_bound=100)
        rna_df['Cell Index'] = indices
        rna_df = rna_df[rna_df['Cell Index'] < centroids.shape[0]]

        # rm other
        rna_df = rna_df[rna_df['Gene'] != 'Other']
        num = len(rna_df['Cell Index'].unique())
        print(f'rm_oth_rna_num:\t{rna_df.shape[0]}')
        print(f'cell_num:\t{num}')
        print(rna_df.loc[:5])

        rna_df.to_csv(input_dir.replace(
            '.csv', '_preprocessed.csv'), index=False)

    return rna_df


def create_exp_matrix(rna_df):
    match_df = rna_df.copy()
    match_df['Count'] = np.ones(len(match_df))
    match_df_group = match_df[['Cell Index', 'Gene', 'Count']].groupby(
        ['Cell Index', 'Gene']).count()
    matrix = match_df_group.unstack().fillna(0)
    matrix.columns = matrix.columns.droplevel()
    matrix.columns.name = 'Gene'
    matrix.index.name = 'Cell Index'
    return matrix


def expression_matrix():
    # read cell centroid
    centroids = pd.read_csv(seg_dir/'dapi_predict.csv',
                            index_col=0).to_numpy(dtype=np.float64)
    centroids[:, 0] *= 3.36
    centroids[:, 1] *= 1
    centroids[:, 2] *= 1
    print(f'centroid_num:\t{len(centroids)}')

    # rad rna information
    rna_df = load_rnas(centroids=centroids, input_dir=read_dir /
                       'mapped_genes.csv', preprocess=False)

    # ## replace gene names
    # PRISM_list = [f'PRISM_{_}' for _ in range(1,31)]
    # gene_order_list = ['Gapdh','Slc1a3', 'Slc17a7', 'Snap25',
    #             'Rasgrf2','Rgs4', 'Prox1', 'Plcxd2', 'Vxn', 'Pcp4', 'Nr4a2', 'Ctgf',
    #             'Gad1', 'Gad2', 'Pvalb', 'Sst', 'Vip', 'Lamp5',
    #             'Aqp4', 'Apod', 'Plp1', 'Cx3cr1', 'Pmch', 'Gfap',
    #             'Cck', 'Mbp', 'Rprm', 'Enpp2', 'Nov', 'Rorb',
    #             ]

    # gene_list = list(pd.read_csv(r'E:\TMC\cell_typing\dataset_spatial\PRISM_mousebrain\var.csv')['gene_names'])
    # replace = {PRISM_list[_]:gene_list[_] for _ in range(len(PRISM_list))}
    # rna_df['Gene'] = rna_df['Gene'].replace(replace)
    # rna_df['Gene'] = rna_df['Gene'].replace({'3110035E14Rik':'Vxn'})
    # rna_df['Gene'] = pd.Categorical(rna_df['Gene'], categories=gene_order_list, ordered=True)

    matrix = create_exp_matrix(rna_df)
    matrix.to_csv(seg_dir/'expression_matrix.csv')


if __name__ == '__main__':
    expression_matrix()
