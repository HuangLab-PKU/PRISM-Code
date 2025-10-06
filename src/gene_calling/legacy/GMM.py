import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path
import shutil
import yaml

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'Arial',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})

package_path = r'C:\Users\Mingchuan\Huanglab\PRISM\PRISM_Code\gene_calling'
if package_path not in sys.path: sys.path.append(package_path)

def apply_gmm(reduced_features, num_clusters, means_init=None):
    gmm = GMM(n_components=num_clusters, covariance_type='diag', means_init=means_init)
    gmm_clusters = gmm.fit(reduced_features)
    labels = gmm_clusters.predict(reduced_features)
    return  gmm_clusters, labels

# workdir 
BASE_DIR = Path(r'G:\spatial_data\processed')
RUN_ID = '20250625_PJR_WSY_Huh7-16'
src_dir = BASE_DIR / f'{RUN_ID}'
read_dir = src_dir / 'readout'
figure_dir = read_dir / 'figures'
read_dir.mkdir(exist_ok=True)
figure_dir.mkdir(exist_ok=True)

# copy the current python file to the read_dir
shutil.copy(os.path.abspath(__file__), read_dir)

# parameters
with open(read_dir / 'params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.UnsafeLoader)

# basic
PRISM_PANEL = params["PRISM_PANEL"]        
GLAYER = params["GLAYER"]                  
COLOR_GRADE = params["COLOR_GRADE"]        
Q_CHNS = params["Q_CHNS"]                  
Q_NUM = params["Q_NUM"]                    
# thresholds
thre_min = params["thre_min"]              
thre_max = params["thre_max"]              
# visualization
XRANGE = params["XRANGE"]                  
YRANGE = params["YRANGE"]                  
s = params["s"]                            
alpha = params["alpha"]                    
percentile_thre = params["percentile_thre"]
bins = tuple(params["bins"])               
# GMM
CD_1_proj = np.array(params["CD_1_proj"])  
CD_2_proj = np.array(params["CD_2_proj"])  
centroid_init_dict = {int(k): v for k, v in params["centroid_init_dict"].items()}
colormap = params["colormap"]

# load data
intensity = pd.read_csv(read_dir / 'intensity_preprocessed.csv')

intensity['label'] = -1
GMM_dict = dict()
for layer in range(GLAYER):
    centroids_init = centroid_init_dict[layer]
    filtered_data = intensity[intensity['G_layer'] == layer]
    reduced_features = filtered_data[Q_CHNS]
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
    
    ax_gmm.scatter(data['CD_1_blur'], data['CD_2_blur'], label=data['label'],
                   c=[colormap[layer][label] for label in data['label']], marker='.', alpha=alpha, s=s)
    for i in range(1 + layer * Q_NUM, 1 + (layer+1) * Q_NUM):
        cen_tmp = np.mean(data[data['label']==i][['CD_1_blur', 'CD_2_blur']], axis=0)
        ax_gmm.text(cen_tmp[0], cen_tmp[1], i, fontsize=12, color='black', ha='center', va='center')
    RYB_xy_transform = np.concatenate([CD_1_proj, CD_2_proj], axis=1)
    centroid_init = centroid_init_dict[layer] @ RYB_xy_transform
    ax_gmm.scatter(centroid_init[:, 0], centroid_init[:, 1], color='cyan', s=1.5, alpha=0.7)

    ax_gmm.set_title(f'G={layer}')
    ax_gmm.set_xlim(XRANGE)
    ax_gmm.set_ylim(YRANGE)

    x, y = data['CD_1_blur'], data['CD_2_blur']
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

# save intensity with gmm label
intensity.to_csv(read_dir / 'intensity_labeled.csv', index=False)