import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})
import seaborn as sns

# add package path
# !the package path should be your_src_path/PRISM_Code/gene_calling/lib
package_path = r'E:\TMC\PRISM_Code\gene_calling\lib'
if package_path not in sys.path: sys.path.append(package_path)

##############
# parameters #
##############

BASE_DIR = Path(r'F:\spatial_data\processed')
RUN_ID = '20240626_FFPE_PRISM30_HCC_CA_5um_SudanBB'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout' / 'GMM'
figure_dir = read_dir / 'figures'
os.makedirs(read_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# Load data and preprocess
PRISM_PANEL = 'PRISM31' # 'PRISM30', 'PRISM31', 'PRISM63', 'PRISM64'
GLAYER = 2
COLOR_GRADE = 5
Q_CHANNELS = ['Ye/A', 'B/A', 'R/A',]
Q_NUM = int(COLOR_GRADE * (COLOR_GRADE + 1)/2)

# visualization
XRANGE = [-0.8, 0.8]
YRANGE = [-0.6, 0.8]

# hist2d of density
s = 0.009
alpha = 0.08
percentile_thre = 99.5
bins = (500, 500)

######################################
# load data, preprocess and overview #
######################################

# load data
intensity_raw = pd.read_csv(read_dir / 'intensity_deduplicated.csv', index_col=0)
intensity = intensity_raw.copy()

## Overview of data distribution
intensity['sum'] = intensity['Scaled_R'] + intensity['Scaled_Ye'] + intensity['Scaled_B']
intensity['Ye/A'] = intensity['Scaled_Ye'] / intensity['sum']
intensity['B/A'] = intensity['Scaled_B'] / intensity['sum']
intensity['R/A'] = intensity['Scaled_R'] / intensity['sum']
intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']

# adjust of G_channel
intensity['G/A'] = np.log(1 + intensity['G/A']) / np.log(10)
if PRISM_PANEL in ('PRISM64', 'PRISM63'):
    intensity['G/A'] = np.log(1 + intensity['G/A']) / np.log(10)
    intensity['G/A'] = intensity['G/A'] * 3
intensity['G/A'] = intensity['G/A'] * np.exp(0.6 * intensity['Ye/A'])

intensity_G = intensity[intensity['G/A'].isna()]
intensity = intensity[~intensity.index.isin(intensity_G.index)]

if PRISM_PANEL in ('PRISM31', 'PRISM64'):
    if PRISM_PANEL == 'PRISM31': THRE = 1.5
    elif PRISM_PANEL == 'PRISM64': THRE = 1.5
    intensity_G = pd.concat([intensity_G, intensity[intensity['G/A'] > THRE]])
    intensity = intensity[~intensity.index.isin(intensity_G.index)]
    intensity_G

RYB_x_transform = np.array([[-np.sqrt(2)/2], [np.sqrt(2)/2], [0]])
RYB_y_transform = np.array([[-1/2], [-1/2], [np.sqrt(2)/2]])
RYB_xy_transform = np.concatenate([RYB_x_transform, RYB_y_transform], axis=1)
intensity['X_coor'] = intensity[Q_CHANNELS] @ RYB_x_transform
intensity['Y_coor'] = intensity[Q_CHANNELS] @ RYB_y_transform

data = intensity.sample(min(10000, len(intensity)))
fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7,5))
ax[0].hist(bins=100, x=data['Ye/A'])
ax[1].hist(bins=100, x=data['B/A'])
ax[2].hist(bins=100, x=data['R/A'])
ax[3].hist(bins=100, x=data['G/A'])
plt.savefig(figure_dir / 'histogram_raw.png', dpi=300, bbox_inches='tight')
plt.close()

## Gaussian blur and orthogonal decomposition
# blur at position 0
gaussian = np.concatenate([np.random.normal(loc=0, scale=0.01, size=intensity[Q_CHANNELS].shape), 
                           np.random.normal(loc=0, scale=0.01, size=intensity[['G/A']].shape)], axis=1)
intensity[Q_CHANNELS + ['G/A']] = intensity[Q_CHANNELS + ['G/A']].mask(intensity[Q_CHANNELS + ['G/A']]==0, gaussian)

# blur at position 1
gaussian = np.random.normal(loc=0, scale=0.01, size=intensity[Q_CHANNELS].shape)
intensity[Q_CHANNELS] = intensity[Q_CHANNELS].mask(intensity[Q_CHANNELS]==1, 1 + gaussian)

intensity['X_coor_gaussian'] = intensity[Q_CHANNELS] @ RYB_x_transform
intensity['Y_coor_gaussian'] = intensity[Q_CHANNELS] @ RYB_y_transform

## Overview of preprocessed data distribution
from scipy.signal import argrelextrema
def plot_hist_with_extrema(a, ax=None, bins=100, extrema='max', kde_kws={'bw_adjust':0.5}):
    sns.histplot(a, bins=bins, stat='count', edgecolor='white', alpha=1, ax=ax, kde=True, kde_kws=kde_kws)
    y = ax.get_lines()[0].get_ydata()
    if extrema == 'max':
        y = -y
    extrema = [float(_/len(y)*(max(a)-min(a))+min(a)) for _ in argrelextrema(np.array(y), np.less)[0]]
    for subextrema in extrema:
        ax.axvline(x=subextrema, color='r', alpha=0.5, linestyle='--')
    return extrema

data = intensity.sample(min(len(intensity),200000))

fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 7))
plt.setp(ax, xlim=(-0.25, 1.2))
Y_maxima = plot_hist_with_extrema(data['Ye/A'], ax=ax[0], extrema='max', kde_kws={'bw_adjust':1.6})
B_maxima = plot_hist_with_extrema(data['B/A'], ax=ax[1], extrema='max', kde_kws={'bw_adjust':1})
R_maxima = plot_hist_with_extrema(data['R/A'], ax=ax[2], extrema='max', kde_kws={'bw_adjust':1})
G_minima = plot_hist_with_extrema(data['G/A'], ax=ax[3], extrema='min', kde_kws={'bw_adjust':2})
if len(Y_maxima) != COLOR_GRADE: Y_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
if len(B_maxima) != COLOR_GRADE: B_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
if len(R_maxima) != COLOR_GRADE: R_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
plt.savefig(figure_dir / 'histogram.png', dpi=300, bbox_inches='tight')
plt.close()

minima = G_minima.copy()
minima = minima[: GLAYER - 1]
minima.insert(0, intensity['G/A'].min()-0.01)
minima.append(intensity['G/A'].max()+0.01)
intensity['G_layer'] = pd.cut(intensity['G/A'], bins=minima, labels=[_ for _ in range(len(minima)-1)], include_lowest=True, right=False)
def ybrg_to_rgb(ybr, g=0):
    y, b, r = ybr
    red = y + r
    green = 0.9 * y + 0.2 * g
    blue = b
    return ((red, green, blue) / np.max((red, green, blue))).clip(0, 1)
def reorder(array, order='PRISM30'):
    if order == 'PRISM30' or order == 'PRISM31':
        relabel = {1:1, 2:6, 3:10, 4:13, 5:15, 6:14, 7:12, 8:9, 9:5, 10:4, 11:3, 12:2, 13:7, 14:11, 15:8}
    elif order == 'PRISM63' or order == 'PRISM64':
        relabel = {1:1, 2:7, 3:12, 4:16, 5:19, 6:21, 7:20, 8:18, 9:15, 10:11, 11:6, 
                     12:5, 13:4, 14:3, 15:2, 16:8, 17:13, 18:9, 19:17, 20:14, 21:10}
    else:print('Undefined order, use PRISM30 or PRISM63 instead.')
    return np.array([array[relabel[_]-1] for _ in relabel])

# preparation for init centroids
import itertools
centroid_init_dict = dict()
colormap = dict()
fig, ax =  plt.subplots(nrows=3, ncols=GLAYER, figsize=(20, 10))
for layer in range(GLAYER):
    data = intensity[intensity['G_layer'] == layer]
    data = data.sample(min(100000, len(data)))
    ax_tmp = ax if GLAYER < 2 else ax[:, layer]
    ax_tmp[0].set_title(f'G_layer{layer}')
    Y_maxima_tmp = plot_hist_with_extrema(data['Ye/A'], ax=ax_tmp[0], extrema='max', kde_kws={'bw_adjust':0.8})
    B_maxima_tmp = plot_hist_with_extrema(data['B/A'], ax=ax_tmp[1], extrema='max', kde_kws={'bw_adjust':0.9})
    R_maxima_tmp = plot_hist_with_extrema(data['R/A'], ax=ax_tmp[2], extrema='max', kde_kws={'bw_adjust':0.7})
    if len(R_maxima_tmp) != COLOR_GRADE: R_maxima_tmp = R_maxima
    if len(Y_maxima_tmp) != COLOR_GRADE: Y_maxima_tmp = Y_maxima
    if len(B_maxima_tmp) != COLOR_GRADE: B_maxima_tmp = B_maxima
    combinations = itertools.product(range(0, COLOR_GRADE), repeat=3)
    filtered_combinations = filter(lambda x: sum(x) == COLOR_GRADE - 1, combinations)
    centroid_init_dict[layer] = np.array([[Y_maxima_tmp[_[0]], B_maxima_tmp[_[1]], R_maxima_tmp[_[2]],] for _ in filtered_combinations])
    centroid_init_dict[layer] = reorder(centroid_init_dict[layer], order=PRISM_PANEL)
    color_list = [ybrg_to_rgb(_, g=layer/GLAYER) for _ in centroid_init_dict[layer]]
    colormap[layer] = {layer*Q_NUM + i + 1:color_list[i] for i in range(len(color_list))}
plt.savefig(figure_dir / 'histogram_by_layer.png', dpi=300, bbox_inches='tight')
plt.close()

n_rows, n_cols = 2, 2 + GLAYER
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols + 2, 10))
ax[1,0].scatter(intensity['X_coor_gaussian'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[1,0].set_ylim([-0.2,1.4])
ax[0,1].scatter(intensity['Y_coor_gaussian'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[0,1].set_ylim([-0.2,1.4])
ax[0,0].scatter(intensity['X_coor_gaussian'], intensity['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
ax[0,0].set_xlim(XRANGE)
ax[0,0].set_ylim(YRANGE)


for subextrema in minima: ax[1,0].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
for subextrema in minima: ax[0,1].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')

for layer in range(GLAYER):
    ax_scatter = ax[0, 2+layer]
    ax_hist = ax[1, 2+layer]
    sub = intensity[(intensity['sum']>1000)&(intensity['sum']<15000)&(intensity['G_layer']==layer)]
    ax_scatter.set_title(f'G={layer}')
    ax_scatter.scatter(sub['X_coor_gaussian'], sub['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
    ax_scatter.set_xlim(XRANGE)
    ax_scatter.set_ylim(YRANGE)

    x, y = sub['X_coor_gaussian'], sub['Y_coor_gaussian']
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    percentile = np.percentile(hist, percentile_thre)
    ax_hist.hist2d(x, y, bins=bins, vmax=percentile,
                   range=[XRANGE, YRANGE], cmap='inferno')
    ax_hist.set_xlim(XRANGE)
    ax_hist.set_ylim(YRANGE)
    
plt.savefig(figure_dir / 'ColorSpace_overview.png', dpi=300, bbox_inches='tight')
plt.close()
gap_l = 0.04
gap_r = 0
for division in minima[1: GLAYER]: intensity = intensity[(intensity['G/A']<division-gap_l)|(intensity['G/A']>division+gap_r)]
s=0.009
alpha=0.08
percentile_thre = 99.5
bins = (500, 500)

n_rows = 2
n_cols = 2 + GLAYER
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols + 2, 10))
ax[1,0].scatter(intensity['X_coor_gaussian'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[1,0].set_ylim([-0.2,1.4])
ax[0,1].scatter(intensity['Y_coor_gaussian'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[0,1].set_ylim([-0.2,1.4])
ax[0,0].scatter(intensity['X_coor_gaussian'], intensity['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
ax[0,0].set_xlim(XRANGE)
ax[0,0].set_ylim(YRANGE)


for subextrema in minima: ax[1,0].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
for subextrema in minima: ax[0,1].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')

for layer in range(GLAYER):
    ax_scatter = ax[0, 2+layer]
    ax_hist = ax[1, 2+layer]
    sub = intensity[(intensity['sum']>1000)&(intensity['sum']<15000)&(intensity['G_layer']==layer)]
    ax_scatter.set_title(f'G={layer}')
    ax_scatter.scatter(sub['X_coor_gaussian'], sub['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
    ax_scatter.set_xlim(XRANGE)
    ax_scatter.set_ylim(YRANGE)

    x, y = sub['X_coor_gaussian'], sub['Y_coor_gaussian']
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    percentile = np.percentile(hist, percentile_thre)
    ax_hist.hist2d(x, y, bins=bins, vmax=percentile,
                   range=[XRANGE, YRANGE], cmap='inferno')
    ax_hist.set_xlim(XRANGE)
    ax_hist.set_ylim(YRANGE)
plt.savefig(figure_dir / 'ColorSpace_G_thre.png', dpi=300, bbox_inches='tight')
plt.close()


## Projection of density in 3D
from scipy import stats

data = intensity.sample(min(20000, len(intensity)))
x = np.array(data['X_coor_gaussian'])
y = np.array(data['Y_coor_gaussian'])
z = np.array(data['G/A'])
xyz = np.vstack([x,y,z])
density = stats.gaussian_kde(xyz)(xyz) 
idx = density.argsort()
x, y, z, density = x[idx], y[idx], z[idx], density[idx]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=density, cmap=plt.cm.gnuplot2, s=0.05, vmin=0, vmax=5) 
ax.set_zlim3d(-0.1, 0.8)
ax.view_init(elev=30, azim=270)
fig.set_facecolor('black')
ax.set_facecolor('black') 
ax.grid(False) 
ax.set_axis_off()
plt.savefig(figure_dir / '3D_density.png', dpi=300, bbox_inches='tight')
plt.close()


####################################
# GMM clustering and visualization #
####################################

from sklearn.mixture import GaussianMixture as GMM
from scipy.spatial.distance import euclidean


def apply_gmm(reduced_features, num_clusters, means_init=None):
    gmm = GMM(n_components=num_clusters, covariance_type='diag', means_init=means_init)
    gmm_clusters = gmm.fit(reduced_features)
    labels = gmm_clusters.predict(reduced_features)
    return  gmm_clusters, labels

intensity['label'] = -1
GMM_dict = dict()
for layer in range(GLAYER):
    centroids_init = centroid_init_dict[layer]
    filtered_data = intensity[intensity['G_layer'] == layer]
    reduced_features = filtered_data[Q_CHANNELS]
    gmm, gmm_labels = apply_gmm(reduced_features, num_clusters=len(centroids_init), means_init=centroids_init)
    GMM_dict[layer] = gmm 
    intensity.loc[filtered_data.index, 'label'] = gmm_labels + int(layer * Q_NUM + 1)
bins = (500, 500)
percentile_thre = 98

fig, ax = plt.subplots(nrows=2, ncols=GLAYER, figsize=(5.5 * GLAYER, 10))
for layer in range(GLAYER):
    ax_gmm = ax[0] if GLAYER < 2 else ax[0, layer]
    ax_hist = ax[1] if GLAYER < 2 else ax[1, layer]
    data = intensity[intensity['G_layer'] == layer]
    data = data[data['label'] != -1]
    gmm = GMM_dict[layer]
    
    # ax_gmm.scatter(data['X_coor_gaussian'], data['Y_coor_gaussian'], c=data['label'], marker='.', alpha=0.1, s=0.1)
    
    ax_gmm.scatter(data['X_coor_gaussian'], data['Y_coor_gaussian'], label=data['label'],
                   c=[colormap[layer][label] for label in data['label']], marker='.', alpha=alpha, s=s, )
    for i in range(1 + layer * Q_NUM, 1 + (layer+1) * Q_NUM):
        cen_tmp = np.mean(data[data['label']==i][['X_coor_gaussian', 'Y_coor_gaussian']], axis=0)
        ax_gmm.text(cen_tmp[0], cen_tmp[1], i, fontsize=12, color='black', ha='center', va='center')
    RYB_xy_transform = np.concatenate([RYB_x_transform, RYB_y_transform], axis=1)
    centroid_init = centroid_init_dict[layer] @ RYB_xy_transform
    ax_gmm.scatter(centroid_init[:, 0], centroid_init[:, 1], color='cyan', s=1.5, alpha=0.7)

    ax_gmm.set_title(f'G={layer}')
    ax_gmm.set_xlim(XRANGE)
    ax_gmm.set_ylim(YRANGE)

    x, y = data['X_coor_gaussian'], data['Y_coor_gaussian']
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    percentile = np.percentile(hist, percentile_thre)
    ax_hist.hist2d(x, y, bins=bins, vmax=percentile,
                   range=[XRANGE, YRANGE], cmap='inferno')
    ax_hist.set_xlim(XRANGE)
    ax_hist.set_ylim(YRANGE)

axes = ax.flat
for i in range(len(axes)):
    ax = axes[i]
    ax.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(figure_dir / f'{i//GLAYER+1}-layer{i%GLAYER+1}.png', bbox_inches=bbox)

plt.tight_layout()
plt.savefig(figure_dir / 'ColorSpace_GMM.png', dpi=300)
plt.close()


# Quantitative analysis of GMM cluster quality
## cluster points num distribution
from lib.GMM_and_visualization import visualization
data = intensity.copy()
plt.figure(figsize=(Q_NUM * GLAYER / 3, 5))
sns.barplot(x = [cluster_num + 1 for cluster_num in range(Q_NUM * GLAYER)], 
            y = [len(data[data['label']==cluster_num+1]) for cluster_num in range(Q_NUM * GLAYER)])
plt.savefig(figure_dir / 'cluster_size.png', dpi=300, bbox_inches='tight')
plt.close()


## cdf related analysis
from lib.GMM_and_visualization import calculate_cdf_mannual

### cdf_4d
#### calculation
cdf_4d, centroids = calculate_cdf_mannual(intensity, st=0, num_per_layer=GLAYER*Q_NUM, channel=['Ye/A', 'B/A', 'R/A', 'G/A'])

#### evaluation
def plot_mean_accuracy(cdfs_df, X_sub, sample=50, y_line=0.95, total_num=GLAYER*Q_NUM, 
                       out_path=read_dir / 'figures', ax=ax):
    sample = 50
    p_thre_list = [_/sample for _ in range(sample)]
    accuracy = []

    for _, p_thre in tqdm(enumerate(p_thre_list), total=len(p_thre_list), desc='accuracy'):
        overlap = pd.DataFrame()

        for cluster_num in range(1, total_num+1):
            tmp = cdfs_df.loc[X_sub['label'][X_sub['label']==(cluster_num)].index]
            overlap[cluster_num] = (tmp>p_thre).sum(axis=0)/len(tmp)
        add = np.diag(overlap) / overlap.sum(axis=1)
        accuracy.append(add)
        overlap = pd.concat([overlap, pd.DataFrame(add).T], axis=0)

    accuracy = np.array(accuracy)

    # Create data
    x = p_thre_list
    y = np.mean(accuracy,axis=1)

    indices = np.where(np.diff(np.sign(y - y_line)))[0][0]
    x_intercepts = x[indices]
    y_intercepts = y[indices]

    # Plot the first dataset on primary axes
    ax.plot(x, y, 'b-')
    ax.set_xlabel('P_thre')
    ax.set_ylabel('mean', color='b')
    ax.tick_params('y', colors='b')

    ax.axhline(y_line, color='r', linestyle='--', label=f'y = {y_line}')
    for x_i, y_i in zip([x_intercepts], [y_intercepts]):
        plt.plot(x_i, y_i, 'ko')
        plt.text(x_i, y_i, f'({x_i:.2f}, {y_i:.2f})')
    ax.set_title("Confidence")

    return accuracy, x_intercepts, y_intercepts


p_thre_list = [0.1, 0.5]
accuracy = []
corr_method = 'spearman'

fig, ax = plt.subplots(nrows=2, ncols=len(p_thre_list) + 1, figsize=(6 * (len(p_thre_list)+1) , 5 * 2))
cdfs_df = cdf_4d.copy()
X_sub = intensity.copy()
ax_heat = ax[0, -1]
corr_matrix = cdfs_df.corr(method=corr_method)
sns.heatmap(corr_matrix, ax=ax_heat, cmap='coolwarm')
ax_heat.set_title(f'{corr_method}_correlation')
for _, p_thre in tqdm(enumerate(p_thre_list), total=len(p_thre_list), desc='p_thre'):
    overlap = pd.DataFrame()
    for cluster_num in range(1, GLAYER*Q_NUM+1):
        tmp = cdfs_df.loc[X_sub['label'][X_sub['label']==(cluster_num)].index]
        overlap[cluster_num] = (tmp>p_thre).sum(axis=0)/len(tmp)
    add = np.diag(overlap) / overlap.sum(axis=0)
    ax[1, _].bar(add.index, add.values)
    accuracy.append(np.mean(add))
    overlap = pd.concat([overlap, pd.DataFrame(add).T], axis=0)
    ax_tmp = ax[0, _]
    ax_tmp.set_title(f'p_thre = {p_thre}')
    sns.heatmap(overlap, vmin=0, vmax=1, ax=ax_tmp)

accuracy, x_intercepts, y_intercepts = plot_mean_accuracy(cdfs_df, X_sub, sample=100, y_line=0.9, total_num=GLAYER*Q_NUM, 
                                                          out_path=figure_dir/'accuracy.png', ax=ax[-1, -1])
plt.tight_layout()
plt.savefig(figure_dir / 'accuracy.pdf')
plt.close()


# cluster visulization after threshold
## threshold
thre = x_intercepts
thre_index = []
cdfs_df = cdf_4d.copy()
for cluster_num in range(1, GLAYER*Q_NUM+1):
    tmp = cdf_4d.loc[intensity['label'][intensity['label']==(cluster_num)].index]
    tmp = tmp[tmp[cluster_num]>thre]
    thre_index += list(tmp.index)

thre_index.sort()
thre_index = pd.Index(thre_index)
thre_index = thre_index.unique()

print(f'thre={thre}\tpoints_kept: {len(thre_index) / len(intensity_raw) * 100 :.1f}%')

## visualization of threshold data
visualization(intensity.loc[thre_index], G_layer=GLAYER, num_per_layer=Q_NUM, 
              colormap_dict=colormap, bins=[500, 500], percentile_thre=99, 
              out_path_dir=figure_dir / 'ColorSpace_selected.png', label=True)

data = intensity.loc[thre_index]
n_rows, n_cols = 2, 2 - ( - (GLAYER - 1) // 2 )
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols + 1, 10))
ax[1,0].scatter(data['X_coor_gaussian'], data['G/A'], s=s, alpha=alpha, linewidths=None)
ax[0,1].scatter(data['Y_coor_gaussian'], data['G/A'], s=s, alpha=alpha, linewidths=None)
ax[0,0].scatter(data['X_coor_gaussian'], data['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
ax[0,0].set_xlim(XRANGE)
ax[0,0].set_ylim(YRANGE)
for subextrema in minima: ax[1,0].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
for subextrema in minima: ax[0,1].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
for layer in range(GLAYER):
    ax_tmp = ax[(1 + layer) % n_rows, 1 - (-layer // n_rows)]
    ax_tmp.set_title(f'G={layer}')
    sub = data[data['G_layer'] == layer]
    ax_tmp.scatter(sub['X_coor_gaussian'], sub['Y_coor_gaussian'], s=s, alpha=alpha, linewidths=None)
    ax_tmp.set_xlim(XRANGE)
    ax_tmp.set_ylim(YRANGE)
plt.tight_layout()
plt.savefig(figure_dir / 'ColorSpace_overview_selected.png', dpi=300, bbox_inches='tight')

############################################
# visualization of each gene in real space #
############################################
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cmap import Colormap

from lib.projection import interpolate_colors
# from projection import projection_gene

data = intensity.loc[thre_index]
if PRISM_PANEL in ['PRISM64', 'PRISM31']:
    if PRISM_PANEL == 'PRISM64': intensity_G['label'] = 64
    elif PRISM_PANEL == 'PRISM31': intensity_G['label'] = 31
    data = pd.concat([data, intensity_G], axis=0)
gene_list = pd.read_excel(src_dir / 'gene_list.xlsx')['gene'].tolist()
mapped_genes = data[['Y', 'X', 'label']]
mapped_genes['Gene'] = mapped_genes['label'].apply(lambda x: gene_list[x-1])
mapped_genes = mapped_genes[['X', 'Y', 'Gene']]


downsample_factor = 100
projection_yrange = mapped_genes['Y'].min() - 2000, mapped_genes['Y'].max() + 2000 
projection_xrange = mapped_genes['X'].min() - 2000, mapped_genes['X'].max() + 2000
bins = ((projection_xrange[1] - projection_xrange[0]) // downsample_factor, 
        (projection_yrange[1] - projection_yrange[0]) // downsample_factor)
main_plot_width_inch = bins[0] / bins[1] * 10
cbar_width_inch = 0.2  # 你希望的颜色条宽度（单位：英寸）
fig_width_inch = main_plot_width_inch + cbar_width_inch + 1
fig_height_inch = 10

# 计算主图的尺寸，预留颜色条的空间
left_space_inch = (fig_width_inch - main_plot_width_inch) / 2
# Load the 'fire' colormap using the custom Colormap class
cmap_fire_custom = Colormap('imagej:fire')
# Extract the colors from the custom colormap
percentile = 100
colors = [c for c in cmap_fire_custom.iter_colors()]
colors = [colors[_] for _ in range(len(colors) * percentile // 100)]
# Interpolate to create a colormap with 256 colors
num_colors = 256
smooth_colors = interpolate_colors(colors, num_colors)
# Create a new ListedColormap
cmap_fire = ListedColormap(smooth_colors)


def projection_gene(x, y, bins=bins, outpath=None):
    # Creating the 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    percentile_max = np.percentile(hist, 99.98)
    percentile_min = min(max(1, np.percentile(hist, 90)), percentile_max // 4)
    # percentile_min = percentile_max / 3
    # percentile_min = 1

    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    ax = fig.add_axes([left_space_inch/fig_width_inch, 0, main_plot_width_inch/fig_width_inch, 1])
    fig.patch.set_facecolor('black')
    plt.hist2d(x, y, 
                range=[projection_xrange, projection_yrange],
                bins=bins, 
                vmax=percentile_max,
                vmin=percentile_min,
                cmap=cmap_fire,
                )
    
    ax.set_facecolor('black')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linestyle('--')
        spine.set_alpha(0.5)

    ax.tick_params(colors='white', which='both')  # 'both' means both major and minor ticks are affected

    cax = fig.add_axes([1 - cbar_width_inch/fig_width_inch, 0, cbar_width_inch/fig_width_inch, 1])  # 注意这里的坐标系是[左, 下, 宽, 高]
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Counts', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.set_tick_params(labelcolor='white')
    cbar.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f'{round(x,1)}')
    cbar.update_ticks()
    
    ax.set_title(f'PRISM_{gene_order}_{gene_name.upper()}', fontsize=20, fontstyle='italic')
    plt.tight_layout()
    if outpath is None: plt.show()
    else: plt.savefig(os.path.join(outpath, f'PRISM_{gene_order}_{gene_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


### batch export 
out_path = read_dir / 'density'
os.makedirs(out_path, exist_ok=True)
for gene_order in tqdm(range(1, len(gene_list)+1)):
  gene_name = gene_list[gene_order-1]
  points = mapped_genes[mapped_genes['Gene']==gene_name][['X', 'Y']].values
  # Extracting x and y coordinates
  x = points[:, 0]
  y = points[:, 1]
  projection_gene(x, y, outpath=out_path)

# export mapped genes
mapped_genes[['Y', 'X', 'Gene']].to_csv(read_dir / 'mapped_genes.csv')


########################################
# visualization of rays in color space #
########################################
fig = plt.figure(figsize=(10, GLAYER*5))
for i in range(GLAYER):
    tmp = intensity[intensity['G_layer']==i]
    tmp = tmp.sample(min(len(tmp),100000))
    x = tmp['R']
    y = tmp['Ye']
    z = tmp['B']
    
    # 创建3D散点图
    ax1 = fig.add_subplot(GLAYER, 2, 2*i+1, projection='3d')
    scatter = ax1.scatter(x, y, z, c=tmp['label'], alpha=0.05, s=0.1, cmap='prism')
    ax1.set_xlabel('R')
    ax1.set_ylabel('Ye')
    ax1.set_zlabel('B')
    ax1.view_init(30, 45)
    ax1.set_xlim([0, 5000])
    ax1.set_ylim([0, 5000])
    ax1.set_zlim([0, 5000])
    ax1.set_title(f'G={i}')

    # 为第二位的子图设置直方图
    ax2 = fig.add_subplot(GLAYER, 2, 2*i+2)
    for label in np.unique(tmp['label']): 
        sns.histplot(tmp[tmp['label']==label]['sum'], bins=100, alpha=0.05, kde=True, stat='density', edgecolor=None, ax=ax2)
    ax2.set_xlim([500, 5000])
    ax2.set_ylim([0, 0.003])
    ax2.legend(np.unique(tmp['label']))

# plt.tight_layout()
plt.savefig(figure_dir / 'ColorSpace_3d_and_sum_evaluation.png', bbox_inches='tight', dpi=300)
plt.close()