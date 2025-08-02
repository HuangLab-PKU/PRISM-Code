# cell show
import os
from skimage import io, transform, morphology
from tqdm import tqdm
import pandas as pd
import numpy as np
import tifffile
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# plot example
# combine_adata_st = combine_adata[combine_adata.obs.dataset == 'st']
# combine_adata_st.obs.index = [_.replace('-st','') for _ in combine_adata_st.obs.index]
# hulls, type_indices = create_hull(combine_adata_st, clus_obs="leiden", cont_thre=15)
# show_cluster(hulls, type_indices, cluster_list, cluster_colormap=["red"] * 200)


def create_hull(adata, clus_obs="leiden", cont_thre=15, rna_pos=[]):
    factor_data = {"Cell_Index": [str(_) for _ in adata.obs.index], "Cluster": [_ for _ in adata.obs[clus_obs]]}
    factor = pd.DataFrame(factor_data).set_index("Cell_Index")
    cluster_num = np.unique(factor["Cluster"])

    type_indices = {}
    for cell_type in cluster_num: type_indices[cell_type] = list(factor[factor["Cluster"] == cell_type].index)

    hulls = {}
    df_group = rna_pos.groupby("Cell Index")

    for group in tqdm(df_group, desc="hull"):
        coordinates = group[1][["Y", "X"]].values
        if len(coordinates) < cont_thre:
            continue
        try:
            hull = ConvexHull(coordinates)
        except:
            continue
        coordinate_path = np.vstack(
            (coordinates[hull.vertices, 0], coordinates[hull.vertices, 1])
        ).T
        hulls[group[0]] = coordinate_path
    return hulls, type_indices


def show_cluster(hulls, type_indices, cluster_list, COI='', cluster_colormap=["red"] * 200, 
                 ax='', linewidth=0.1, xlim=[0, 37500], ylim=[0, 35000],
                 name='projection', show=True, show_legend=False, save=False, outpath='',
                 leave=True, position=0, verbose=False):
    cell = 0
    for ind in cluster_list:
        for idx in tqdm(type_indices[ind], desc=f'cell for cluster_{ind}', leave=leave, position=position):
            idx = int(idx)
            try:
                ax.fill(hulls[idx][:, 1], hulls[idx][:, 0], 
                        color=cluster_colormap[ind], label=ind,
                        linewidth=linewidth, alpha=1)
                if cluster_colormap[ind]!=(0.9,0.9,0.9): cell += 1
            
            except KeyError: pass
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"cell_num={cell}")
    if show_legend: 
        legend_handles = [Patch(facecolor=cluster_colormap[cluster], label=cluster) for cluster in cluster_colormap]
        ax.legend(handles=legend_handles, loc='upper right')
        
    if verbose: print(f'{cell} cells have been ploted.')
    
    if save: plt.savefig(outpath)
    elif show: plt.show()
    else: pass    


def ROI_mask_load(input_path, out_path, save=False, blur=True):
    ROI_mask = {}
    for mask_file in os.listdir(input_path):
        image = io.imread(os.path.join(input_path, mask_file))
        if blur:
            image = transform.rotate(image, angle=90, resize=True)
            image = morphology.binary_dilation(
                image, footprint=morphology.disk(5))
        ROI_mask[mask_file.replace('.tif', '').replace('Mask', 'ROI')] = image

    ncols = int(-(-len(ROI_mask)**(1/2)//1))
    nrows = -(-len(ROI_mask)//ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols*4, nrows*4))
    for pos, mask_name in enumerate(list(ROI_mask.keys())):
        try:
            ax[pos // ncols][pos % ncols].imshow(ROI_mask[mask_name])
        except:
            ax[pos // ncols][pos % ncols].imshow(ROI_mask[mask_name][0, :, :])
        ax[pos // ncols][pos % ncols].set_title(mask_name)
        ax[pos // ncols][pos % ncols].set_xlabel("")
        ax[pos // ncols][pos % ncols].set_ylabel("")
    fig.suptitle('Mask_of_ROIs', fontsize=20)
    plt.tight_layout()

    if save:
        plt.savefig('{}/{}.png'.format(out_path, 'mask_of_ROIs'))
    else:
        plt.show()

    return ROI_mask


# Example points (Replace this with your actual data)
# Format: [(x1, y1, z1), (x2, y2, z2), ...]
def downsize_to_tif(points, out_path, max_point_values=0, binsize=100):

    # Determine the size of the grid based on the maximum point values
    if max_point_values == 0:
        max_point_values = np.max(points, axis=0)

    grid_size = tuple(
        np.ceil(np.array(max_point_values) / binsize).astype(int))
    points = points.astype(np.uint16) // binsize

    # Create an empty grid
    voxel_grid = np.zeros(grid_size, dtype=np.uint16)

    # Assign points to the voxel grid
    for x, y, z in tqdm(points):
        try:
            voxel_grid[z, y, x] += 1
        except:
            print(z, y, x)

    # Save the grid as a 3D TIF file
    tifffile.imsave(out_path, voxel_grid)
