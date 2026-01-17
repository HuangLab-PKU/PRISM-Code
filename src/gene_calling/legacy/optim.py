import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml

import pandas as pd
import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'Arial',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})

from lib.GMM_and_visualization import color_space_visual

package_path = r'C:\Users\Mingchuan\Huanglab\PRISM\PRISM_Code\gene_calling'
if package_path not in sys.path: sys.path.append(package_path)

# workdir 
BASE_DIR = Path(r'G:\spatial_data\processed')
RUN_ID = '20250318_ZP_YCXin_PRISM_pro_exp_4_con_2_EtOH_omit_primary_antibody'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
read_dir = src_dir / 'readout'
figure_dir = read_dir / 'figures'
read_dir.mkdir(exist_ok=True)
figure_dir.mkdir(exist_ok=True)

# copy the current python file to the read_dir
shutil.copy(os.path.abspath(__file__), read_dir)

# parameters
with open(read_dir/'params.yaml', 'r') as f: params = yaml.load(f, Loader=yaml.UnsafeLoader)
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

# Load data
intensity_raw = pd.read_csv(read_dir / 'tmp' / 'intensity_raw.csv')
intensity = pd.read_csv(read_dir / 'intensity_labeled.csv')
intensity_G = pd.read_csv(read_dir / 'intensity_G.csv')

from lib.manual_thre import relabel
print(len(intensity_raw))
print(len(intensity))
intensity = relabel(intensity, mask_dir=read_dir/'mask_check', mode='replace', num_per_layer=Q_NUM, xrange=XRANGE, yrange=YRANGE)
intensity = relabel(intensity, mask_dir=read_dir/'mask_check', mode='discard', num_per_layer=Q_NUM, xrange=XRANGE, yrange=YRANGE)
print(len(intensity))

# visualization
## spots num
data = intensity.copy()
plt.figure(figsize=(Q_NUM * GLAYER / 3, 5))
sns.barplot(x = [cluster_num + 1 for cluster_num in range(Q_NUM * GLAYER)], 
            y = [len(data[data['label']==cluster_num+1]) for cluster_num in range(Q_NUM * GLAYER)])
plt.savefig(figure_dir / 'cluster_size.png', dpi=300, bbox_inches='tight')
plt.close()

## color space visualization
color_space_visual(intensity, G_layer=GLAYER, num_per_layer=Q_NUM, bins=[500, 500], 
              percentile_thre=99, colormap_dict=colormap, XRANGE=XRANGE, YRANGE=YRANGE,
              out_path_dir=figure_dir / 'ColorSpace.png', label=True)

# Quality control
from lib.quantitative_evaluation import calculate_cdf, plot_mean_accuracy

cdf_4d, centroids = calculate_cdf(intensity, st=0, num_per_layer=GLAYER*Q_NUM, channel=['Ye/A', 'B/A', 'R/A', 'G/A'])
### evaluation
p_thre_list = [0.1, 0.5]
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
    # overlap calculation
    add = np.diag(overlap) / overlap.sum(axis=0)
    ax[1, _].bar(add.index, add.values)
    overlap = pd.concat([overlap, pd.DataFrame(add).T], axis=0)
    ax_tmp = ax[0, _]
    ax_tmp.set_title(f'p_thre = {p_thre}')
    sns.heatmap(overlap, vmin=0, vmax=1, ax=ax_tmp)
accuracy, x_intercepts, y_intercepts = plot_mean_accuracy(cdfs_df, X_sub, sample=100, y_line=0.9, total_num=GLAYER*Q_NUM, ax=ax[-1, -1])
plt.tight_layout()
plt.savefig(figure_dir / 'accuracy.pdf')
plt.close()

### threshold
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
data = intensity.copy()
data = data.loc[thre_index]
if PRISM_PANEL in ['PRISM64', 'PRISM31']:
    if PRISM_PANEL == 'PRISM64': intensity_G['label'] = 64
    elif PRISM_PANEL == 'PRISM31': intensity_G['label'] = 31
    data = pd.concat([data, intensity_G], axis=0)

## deduplicate
# from lib.data_preprocess import deduplicate_df
# # deduplicated
# mapped_genes = pd.DataFrame()
# df = data[['Y','X','label']]
# df = df[df['label']!=0]
# for gene in tqdm(set(df['label'])):
#     df_gene = df[df['label'] == gene]
#     df_gene_reduced = deduplicate_df(df_gene)
#     mapped_genes = pd.concat([mapped_genes, df_gene_reduced])
# print(len(mapped_genes))

## visualization
# spots num
plt.figure(figsize=(Q_NUM * GLAYER / 3, 5))
sns.barplot(x = [cluster_num + 1 for cluster_num in range(Q_NUM * GLAYER)], 
            y = [len(data[data['label']==cluster_num+1]) for cluster_num in range(Q_NUM * GLAYER)])
plt.savefig(figure_dir / 'cluster_size_QC.png', dpi=300, bbox_inches='tight')
plt.close()
# color space
color_space_visual(intensity.loc[thre_index], G_layer=GLAYER, num_per_layer=Q_NUM, 
              colormap_dict=colormap, bins=[500, 500], percentile_thre=99, 
              out_path_dir=figure_dir / 'ColorSpace_QC.png', label=True)

# # vectors
# fig = plt.figure(figsize=(10, GLAYER*5))
# for i in range(GLAYER):
#     tmp = intensity[intensity['G_layer']==i]
#     tmp = tmp.sample(min(100000,len(tmp)))
#     x = tmp['R']
#     y = tmp['Ye']
#     z = tmp['B']
    
#     # 创建3D散点图
#     ax1 = fig.add_subplot(GLAYER, 2, 2*i+1, projection='3d')
#     scatter = ax1.scatter(x, y, z, c=tmp['label'], alpha=0.05, s=0.1, cmap='prism')
#     ax1.set_xlabel('R')
#     ax1.set_ylabel('Ye')
#     ax1.set_zlabel('B')
#     ax1.view_init(30, 45)
#     ax1.set_xlim([0, 5000])
#     ax1.set_ylim([0, 5000])
#     ax1.set_zlim([0, 5000])
#     ax1.set_title(f'G={i}')

#     # 为第二位的子图设置直方图
#     ax2 = fig.add_subplot(GLAYER, 2, 2*i+2)
#     for label in np.unique(tmp['label']): 
#         sns.histplot(tmp[tmp['label']==label]['sum'], bins=100, alpha=0.05, kde=True, stat='density', edgecolor=None, ax=ax2)
#     ax2.set_xlim([500, 5000])
#     ax2.set_ylim([0, 0.003])
#     ax2.legend(np.unique(tmp['label']))
# plt.tight_layout()
# plt.savefig(figure_dir / 'ColorSpace_vertor.png', bbox_inches='tight', dpi=300)
# plt.close()

# Export
gene_info = pd.read_excel(src_dir / 'gene_list.xlsx',index_col=0)
mapped_genes = data[['Y', 'X', 'label']]
gene_info_dict = gene_info['gene_name'].to_dict()
mapped_genes = data[['Y', 'X', 'label']].copy()
mapped_genes['Gene'] = mapped_genes['label'].map(lambda x: gene_info_dict.get(f'PRISM_{x}', 'default_value'))
mapped_genes = mapped_genes[['X', 'Y', 'Gene']]
mapped_genes[['Y', 'X', 'Gene']].to_csv(read_dir / 'mapped_genes.csv')
print(mapped_genes.head())