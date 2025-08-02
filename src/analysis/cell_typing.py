from doctest import OutputChecker
import pickle
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import scanpy as sc
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# Preprocessing
def adata_filter(adata, min_genes, min_counts, max_counts, min_cells):
    adata_filtered = adata.copy()
    sc.pp.filter_cells(adata_filtered, min_genes=min_genes)
    sc.pp.filter_cells(adata_filtered, min_counts=min_counts)
    sc.pp.filter_cells(adata_filtered, max_counts=max_counts)
    sc.pp.filter_genes(adata_filtered, min_cells=min_cells)
    return adata_filtered


def QC_plot(adata, hue, min_counts='nan', max_counts='nan', min_genes='nan', min_cells='nan'):
    g = sns.JointGrid(
        data=adata.obs,
        x="total_counts",
        y="n_genes_by_counts",
        height=5,
        ratio=2,
        hue=hue,
    )

    g.plot_joint(sns.scatterplot, s=40, alpha=0.3)
    g.plot_marginals(sns.kdeplot)
    g.set_axis_labels("total_counts", "n_genes_by_counts", fontsize=16)
    g.fig.set_figwidth(6)
    g.fig.set_figheight(6)
    g.fig.suptitle("QC_by_{}, cell_num={}, gene_num={}\n\
                   min_counts={}, max_counts={}, min_genes={}, min_cells={}\
                   \n\n\n\n\n".format(hue, len(adata), len(adata.var.index), min_counts, max_counts, min_genes, min_cells))
    plt.show()


def general_preprocess(adata, min_genes=2, min_counts=5, max_counts=200, min_cells=3, auto_filter=False, hue='dataset'):
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    # You can adjust the overall figure size here
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3)
    # Plot top 20 most expressed genes
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.highest_expr_genes(adata, n_top=10, ax=ax1, show=False)
    # distribution of cell counts
    ax2 = fig.add_subplot(gs[0, 1:3])
    counts = adata.obs.total_counts
    sns.histplot(counts, stat='count', ax=ax2,
                 bins=150, edgecolor='white', linewidth=0.5, alpha=1,
                 kde=True, line_kws=dict(color='black', alpha=0.7, linewidth=1.5, label='KDE'), kde_kws={'bw_adjust': 1},
                 )
    y = ax2.get_lines()[0].get_ydata()
    maxima = [float(_/len(y)*(max(counts)-min(counts))+min(counts))
              for _ in argrelextrema(-np.array(y), np.less)[0]]
    print(f'maxima: {maxima}')
    plt.tight_layout()
    plt.show()
    plt.close(fig=fig)

    # plot origin and filtered in a combined figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 6)
    categories = adata.obs[hue].unique()

    ax_1_scatter = fig.add_subplot(gs[1:3, 0:2])
    sns.scatterplot(x=adata.obs.total_counts, y=adata.obs.n_genes_by_counts,
                    hue=adata.obs[hue], ax=ax_1_scatter, )

    ax_1_count = fig.add_subplot(gs[0:1, 0:2])
    for category in categories:
        subset = adata[adata.obs[hue] == category]
        sns.kdeplot(subset.obs.total_counts, ax=ax_1_count)
    ax_1_count.xaxis.set_visible(False)
    ax_1_count.yaxis.set_visible(False)
    ax_1_count.grid(False)

    ax_1_gene = fig.add_subplot(gs[1:3, 2:3])
    for category in categories:
        subset = adata[adata.obs[hue] == category]
        sns.kdeplot(y=subset.obs.n_genes_by_counts, ax=ax_1_gene)
    ax_1_gene.xaxis.set_visible(False)
    ax_1_gene.yaxis.set_visible(False)
    ax_1_gene.grid(False)

    ax_1_count.set_title("QC_by_{}, cell_num={}, gene_num={}\n\
                         min_counts={}, max_counts={}, \n\
                         min_genes={}, min_cells={}\
                         ".format(hue, len(adata), len(adata.var.index), 'nan', 'nan', 'nan', 'nan'))

    origin_cell_num = len(adata)
    min_counts = int(maxima[0]) if auto_filter else min_counts
    max_counts = int(np.percentile(counts, 99.9)
                     ) if auto_filter else max_counts
    adata_filtered = adata_filter(
        adata, min_genes, min_counts, max_counts, min_cells)
    filtered_cell_num = len(adata_filtered)

    ax_2_scatter = fig.add_subplot(gs[1:3, 3:5])
    sns.scatterplot(x=adata_filtered.obs.total_counts, y=adata_filtered.obs.n_genes_by_counts,
                    hue=adata_filtered.obs[hue], ax=ax_2_scatter, )

    ax_2_count = fig.add_subplot(gs[0:1, 3:5])
    categories = adata_filtered.obs[hue].unique()
    for category in categories:
        subset = adata_filtered[adata_filtered.obs[hue] == category]
        sns.kdeplot(subset.obs.total_counts, ax=ax_2_count)
    ax_2_count.xaxis.set_visible(False)
    ax_2_count.yaxis.set_visible(False)
    ax_2_count.grid(False)

    ax_2_gene = fig.add_subplot(gs[1:3, 5:6])
    for category in categories:
        subset = adata_filtered[adata_filtered.obs[hue] == category]
        sns.kdeplot(y=subset.obs.n_genes_by_counts, ax=ax_2_gene)
    ax_2_gene.xaxis.set_visible(False)
    ax_2_gene.yaxis.set_visible(False)
    ax_2_gene.grid(False)

    ax_2_count.set_title("QC_by_{}, cell_num={}, gene_num={}\n\
                         min_counts={}, max_counts={}, \n\
                         min_genes={}, min_cells={}\
                         ".format(hue, len(adata_filtered), len(adata_filtered.var.index), min_counts, max_counts, min_genes, min_cells))

    plt.tight_layout()

    plt.show()
    plt.close(fig=fig)

    # plot origin

    # QC_plot(adata, hue='dataset')

    # # plot filtered
    # min_counts = int(maxima[0]) if auto_filter else min_counts
    # max_counts = int(np.percentile(counts, 99.9)) if auto_filter else max_counts
    # adata = adata_filter(adata, min_genes, min_counts, max_counts, min_cells)
    # filtered_cell_num = len(adata)
    # QC_plot(adata, hue='dataset', min_genes=min_genes, min_counts=min_counts, max_counts=max_counts, min_cells=min_cells)
    return adata_filtered, origin_cell_num, filtered_cell_num


# g = sns.JointGrid(
#     data=adata.obs,
#     x="total_counts",
#     y="n_genes_by_counts",
#     height=5,
#     ratio=2,
#     hue=hue,
# )

# g.plot_joint(sns.scatterplot, s=40, alpha=0.3)
# g.plot_marginals(sns.kdeplot)
# g.set_axis_labels("total_counts", "n_genes_by_counts", fontsize=16)
# g.fig.set_figwidth(6)
# g.fig.set_figheight(6)
# g.fig.suptitle("QC_by_{}, cell_num={}, gene_num={}\n\
#                 min_counts={}, max_counts={}, min_genes={}, min_cells={}\
#                 \n\n\n\n\n".format(hue,len(adata),len(adata.var.index),min_counts, max_counts, min_genes, min_cells))
# plt.show()


def preprocess_of_UMAP(adata):
    # Normalization scaling
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    # Scale data to unit variance and zero mean
    sc.pp.regress_out(adata, ["total_counts"])
    sc.pp.scale(adata)
    return adata


def save_pos_on_UMAP(adata, out_dir):
    try:
        adata_coor = pd.DataFrame(adata.obsm["X_umap"], columns=[
                                  "Coor_X", "Coor_Y"], index=adata.obs.index)
        df = pd.concat([adata_coor["Coor_X"], adata_coor["Coor_Y"],
                        pd.DataFrame(adata.obs.index), adata.obs.leiden], axis=1)
        df.to_csv(out_dir)
    except KeyError:
        print('X_umap not found, please perform umap first.')


def save_cell_cluster(adata, out_path, st_point, cell_num, name="leiden"):
    raw_clu = dict(adata.obs[name])
    cluster = dict()
    for cell_num in raw_clu.keys():
        cluster[cell_num] = -1

    for cell in raw_clu.keys():
        cluster[int(cell) - st_point] = int(raw_clu[cell])

    with open(out_path, "wb") as handle:
        pickle.dump(cluster, handle)


def UMAP_genes_plot(adata, gene_list=None, size=0.1, vmin=0, vmax=5, out_path=None):
    n_pcs = len(adata.uns['pca']['variance'])
    n_neighbors = adata.uns['neighbors']['params']['n_neighbors']
    if gene_list == None:
        gene_list = list(adata.var_names)

    # Plot Gene distribution
    ncols = int(-(-len(adata.var_names)**(1/2)//1))
    nrows = -(-len(adata.var_names)//ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols*4, nrows*4))
    for pos, gene_name in enumerate(gene_list):
        sc.pl.umap(
            adata,
            vmin=vmin, vmax=vmax, size=size, color=gene_name,
            legend_fontweight=100, legend_fontsize=20,
            show=False, ax=ax[pos // ncols][pos % ncols],
        )
        ax[pos // ncols][pos % ncols].set_xticklabels("")
        ax[pos // ncols][pos % ncols].set_yticklabels("")

    fig.suptitle(
        "{}\nexp:{}\nUMAP:{}\n".format(
            f"UMAP_by_genes",
            f"cell_num={len(adata)}",
            f"n_neighbors={n_neighbors}, n_pcs={n_pcs}"),
        fontsize=20,)
    plt.tight_layout()

    if out_path == None:
        plt.show()
    else:
        plt.savefig(os.path.join(out_path, "UMAP_genes.png"), dpi=300, bbox_inches='tight')
        plt.close()


def UMAP_obs_plot(adata, color='leiden', devided_by='dataset', devide_list=['PRISM'], 
                  face_color='white', legend_loc='on data', palette=False, size=1, out_path=None, ):
    n_pcs = len(adata.uns['pca']['variance'])
    n_neighbors = adata.uns['neighbors']['params']['n_neighbors']
    resolution = adata.uns['leiden']['params']['resolution']

    # Plot Cluster
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    if palette != None:
        sc.pl.umap(adata[adata.obs[devided_by].isin(devide_list)], size=size, color=color, palette=palette,
                   legend_loc=legend_loc, legend_fontsize=7, ax=ax[0], show=False)
        sc.pl.umap(adata, size=size, color=devided_by, ax=ax[1], show=False,
                   legend_fontweight=100, legend_fontsize=20)
        ax[0].set_facecolor(face_color)
        ax[1].set_facecolor(face_color)
        if legend_loc == 'right margin':
            # 获取当前的图例并移除
            legend = ax[0].get_legend()
            legend.remove()

            by_label = {label: idx for idx, label in enumerate(adata.obs[color].cat.categories)}
            legend_handles = [Patch(color=palette[label], label=label) for label in by_label]
            legend = ax[0].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    else:
        sc.pl.umap(adata[adata.obs.dataset.isin(datasets)], size=size, color=color,
                   legend_loc=legend_loc, legend_fontsize=7, ax=ax[0], show=False)
        sc.pl.umap(adata, size=size, color="dataset", ax=ax[1], show=False,
                   legend_fontweight=100, legend_fontsize=20)
        ax[0].set_facecolor(face_color)
        ax[1].set_facecolor(face_color)
        if legend_loc == 'right margin':
            # 获取当前的图例并移除
            legend = ax[0].get_legend()
            legend.remove()

            by_label = {label: idx for idx, label in enumerate(adata.obs[color].cat.categories)}
            legend_handles = [Patch(color=palette[label], label=label) for label in by_label]
            legend = ax[0].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    fig.suptitle(
        "{}\nexp:{}\nUMAP:{}\n".format(
            f"UMAP_by_{color}",
            f"cell_num={len(adata)}",
            f"n_neighbors={n_neighbors}, n_pcs={n_pcs}, resolution={resolution}"),
        fontsize=20,
    )

    if out_path != None:
        plt.savefig(os.path.join(out_path, f'UMAP_{color}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def leiden_QC_plot(adata, color='leiden'):
    # cluster QC
    g = sns.JointGrid(
        data=adata.obs,
        x="total_counts",
        y="n_genes_by_counts",
        height=5,
        ratio=2,
        hue=color,
    )
    g.plot_joint(sns.scatterplot, s=40, alpha=0.3)
    g.plot_marginals(sns.kdeplot)
    g.set_axis_labels("total_counts", "n_genes_by_counts", fontsize=8)
    g.fig.set_figwidth(3)
    g.fig.set_figheight(3)
    plt.show()


def annotate(adata, cluster_dict, in_leiden='tmp_leiden', out_leiden='new_leiden', out_type='type'):
    adata.obs[out_leiden] = ["-2"] * len(adata)
    adata.obs[out_type] = ["other"] * len(adata)
    for cluster_num, cluster_name in enumerate(cluster_dict.keys()):
        for sub_cluster in cluster_dict[cluster_name]:
            temp = adata[adata.obs[in_leiden] == str(sub_cluster)]
            adata.obs[out_leiden][temp.obs.index] = [
                str(cluster_num)] * len(temp)
            adata.obs[out_type][temp.obs.index] = [
                str(cluster_name)] * len(temp)
    temp = adata[adata.obs[out_leiden] == '-2']
    adata.obs[out_type][temp.obs.index] = ['other'] * len(temp)
    return adata
