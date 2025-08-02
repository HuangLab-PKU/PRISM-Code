import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path
import shutil
import yaml

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'Arial',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
})

# 添加PRISM代码路径
package_path = r'path_to_PRISM_code_src'
if package_path not in sys.path: sys.path.append(package_path)

# workdir 
BASE_DIR = Path(r'path_to_processed_dataset')
RUN_ID = 'example_data'
src_dir = BASE_DIR / f'{RUN_ID}'
read_dir = src_dir / 'readout'
figure_dir = read_dir / 'figures'
read_dir.mkdir(exist_ok=True)
figure_dir.mkdir(exist_ok=True)

# copy the current python file to the read_dir
shutil.copy(os.path.abspath(__file__), read_dir)

# parameters
## basic
PRISM_PANEL = 'PRISM30' # 'PRISM30', 'PRISM31', 'PRISM45', 'PRISM46', 'PRISM63', 'PRISM64'
GLAYER, COLOR_GRADE = 2, 5
Q_CHNS = ['Ye/A', 'B/A', 'R/A']
Q_NUM = int(COLOR_GRADE * (COLOR_GRADE + 1)/2)
## sum intensity threshold
thre_min, thre_max = 200, 10000
## visualization
XRANGE, YRANGE = [-0.8, 0.8], [-0.6, 0.8]
s = 0.05
alpha = 0.05
percentile_thre = 99.8
bins = (500, 500)


## Data Loading
# intensity_raw = pd.read_csv(read_dir / 'intensity_deduplicated.csv', index_col=0)
# intensity = intensity_raw.copy()
# intensity.head()
intensity_raw = pd.read_csv(read_dir / 'tmp' / 'intensity_raw.csv', index_col=0)
intensity = intensity_raw.copy()

# crosstalk elimination
intensity['B'] = intensity['B'] - intensity['G'] * 0.25
intensity['B'] = np.maximum(intensity['B'], 0)

# scale the intensity
intensity['Scaled_R'] = intensity['R']
intensity['Scaled_Ye'] = intensity['Ye']
intensity['Scaled_G'] = intensity['G'] * 2.5
intensity['Scaled_B'] = intensity['B'] * 0.75

# threshold by intensity
intensity['sum'] = intensity['Scaled_R'] + intensity['Scaled_Ye'] + intensity['Scaled_B']

# normalize
intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']
intensity.loc[intensity['G/A']>5, 'G/A'] = 5
# intensity.head()

# Preprocess
# set 1 subplot on the above and 4 on the below
# the above subplot is the histogram of intensity sum column
# the below subplots are the histogram of each channel
# use gridspec to make the 1 row one subplot

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
ax0 = plt.subplot(gs[0,:])
sns.histplot(intensity['sum'], bins=400, ax=ax0)
ax0.set_xlim((0, 20000))
ax0.set_title('Intensity Sum')

for i, channel in enumerate(['R','Ye','B','G']):
    ax = plt.subplot(gs[1, i])
    sns.histplot(intensity.loc[intensity[channel]>0, channel], bins=300, ax=ax)
    median = intensity.loc[intensity[channel]>0, channel].median()
    ax.set_xlim((0, 10000))
    ax.set_title(f'{channel}_{median:.0f}')

plt.tight_layout()
plt.savefig(figure_dir / 'hist_intensity.png')
plt.close()


# deduplicate
from lib.data_preprocess import deduplicate_df
intensity = intensity[(intensity['sum']>thre_min)&(intensity['sum']<thre_max)]
intensity = deduplicate_df(intensity, columns=['Y','X'], sort_by='sum', threshold=2)

## Raw Data distribution
intensity['Ye/A'] = intensity['Scaled_Ye'] / intensity['sum']
intensity['B/A'] = intensity['Scaled_B'] / intensity['sum']
intensity['R/A'] = intensity['Scaled_R'] / intensity['sum']
intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']
intensity_G = intensity[intensity['G/A'].isna()]
intensity = intensity[~intensity.index.isin(intensity_G.index)]

if PRISM_PANEL in ('PRISM31', 'PRISM46', 'PRISM64'):
    if PRISM_PANEL == 'PRISM31': THRE = 3
    elif PRISM_PANEL == 'PRISM46': THRE = 3
    elif PRISM_PANEL == 'PRISM64': THRE = 3
    intensity_G = pd.concat([intensity_G, intensity[intensity['G/A'] > THRE]])
    intensity = intensity[~intensity.index.isin(intensity_G.index)]

# adjust of FRET
intensity['G/A'] = intensity['G/A'] * np.exp(0.6 * intensity['Ye/A'])
intensity['B/A'] = intensity['B/A'] * np.exp(0.1 * intensity['G/A'])

# adjust of G_channel
intensity['G/A'] = np.log(1 + intensity['G/A']) / np.log(10)
if PRISM_PANEL in ('PRISM64', 'PRISM63'):
    intensity['G/A'] = np.log(1 + intensity['G/A']) / np.log(10)
    intensity['G/A'] = intensity['G/A'] * 3
intensity['G/A'] = np.log1p(intensity['G/A']) 

# sum 1
intensity['sum_scale'] = intensity['R/A'] + intensity['Ye/A'] + intensity['B/A']
intensity['R/A'] = intensity['R/A'] / intensity['sum_scale']
intensity['Ye/A'] = intensity['Ye/A'] / intensity['sum_scale']
intensity['B/A'] = intensity['B/A'] / intensity['sum_scale']

CD_1_proj = np.array([[-np.sqrt(2)/2], [np.sqrt(2)/2], [0]])
CD_2_proj = np.array([[-1/2], [-1/2], [np.sqrt(2)/2]])
RYB_xy_transform = np.concatenate([CD_1_proj, CD_2_proj], axis=1)
intensity['CD_1'] = intensity[Q_CHNS] @ CD_1_proj
intensity['CD_2'] = intensity[Q_CHNS] @ CD_2_proj
data = intensity.copy()
fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7,5))
ax[0].hist(bins=100, x=data['Ye/A'])
ax[1].hist(bins=100, x=data['B/A'])
ax[2].hist(bins=100, x=data['R/A'])
ax[3].hist(bins=100, x=data['G/A'])
plt.savefig(figure_dir / 'hist_raw.png', dpi=300, bbox_inches='tight')
plt.close()

## Gauss-blur and Orth-decompos
# blur at position 0
gaussian = np.concatenate([np.random.normal(loc=0, scale=0.01, size=intensity[Q_CHNS].shape), np.random.normal(loc=0, scale=0.01, size=intensity[['G/A']].shape)], axis=1)
intensity[Q_CHNS + ['G/A']] = intensity[Q_CHNS + ['G/A']].mask(intensity[Q_CHNS + ['G/A']]==0, gaussian)
# blur at position 1
gaussian = np.random.normal(loc=0, scale=0.01, size=intensity[Q_CHNS].shape)
intensity[Q_CHNS] = intensity[Q_CHNS].mask(intensity[Q_CHNS]==1, 1 + gaussian)
intensity['CD_1_blur'] = intensity[Q_CHNS] @ CD_1_proj
intensity['CD_2_blur'] = intensity[Q_CHNS] @ CD_2_proj

## Preprocessed data distribution
### hist by channel
from scipy.signal import argrelextrema
def plot_hist_with_extrema(a, ax=None, bins=100, extrema='max', kde_kws={'bw_adjust':0.5}):
    sns.histplot(a, bins=bins, stat='count', edgecolor='white', alpha=1, ax=ax, kde=True, kde_kws=kde_kws)
    y = ax.get_lines()[0].get_ydata()
    if extrema == 'max': y = -y
    extrema = [float(_/len(y)*(max(a)-min(a))+min(a)) for _ in argrelextrema(np.array(y), np.less)[0]]
    for subextrema in extrema: ax.axvline(x=subextrema, color='r', alpha=0.5, linestyle='--')
    return extrema

data = intensity.sample(min(len(data), 200000))
fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 7))
plt.setp(ax, xlim=(-0.25, 1.2))
Y_maxima = plot_hist_with_extrema(data['Ye/A'], ax=ax[0], extrema='max', kde_kws={'bw_adjust':1.6})
B_maxima = plot_hist_with_extrema(data['B/A'], ax=ax[1], extrema='max', kde_kws={'bw_adjust':1})
R_maxima = plot_hist_with_extrema(data['R/A'], ax=ax[2], extrema='max', kde_kws={'bw_adjust':1})
G_minima = plot_hist_with_extrema(data['G/A'], ax=ax[3], extrema='min', kde_kws={'bw_adjust':2})
if len(Y_maxima) != COLOR_GRADE: Y_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
if len(B_maxima) != COLOR_GRADE: B_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
if len(R_maxima) != COLOR_GRADE: R_maxima = [(_) / (COLOR_GRADE-1) for _ in range(COLOR_GRADE)]
plt.savefig(figure_dir / 'hist_chn.png', dpi=300, bbox_inches='tight')
plt.close()

minima = G_minima.copy()
minima = minima[: GLAYER - 1]
minima.insert(0, intensity['G/A'].min()-0.01)
minima.append(intensity['G/A'].max()+0.01)
intensity['G_layer'] = pd.cut(intensity['G/A'], bins=minima, labels=[_ for _ in range(len(minima)-1)], include_lowest=True, right=False)

### hist by Glayer
# preparation for init centroids
import itertools
def ybrg_to_rgb(ybr, g=0):
    y, b, r = ybr
    red = y + r
    green = 0.9 * y + 0.2 * g
    blue = b
    return ((red, green, blue) / np.max((red, green, blue))).clip(0, 1)
def reorder(array, order='PRISM30'):
    if order in ('PRISM30', 'PRISM31', 'PRISM45', 'PRISM46'): 
        relabel = {1:1, 2:6, 3:10, 4:13, 5:15, 6:14, 7:12, 8:9, 9:5, 10:4, 11:3, 12:2, 13:7, 14:11, 15:8}
    elif order == ('PRISM63', 'PRISM64'):
        relabel = {1:1, 2:7, 3:12, 4:16, 5:19, 6:21, 7:20, 8:18, 9:15, 10:11, 11:6, 12:5, 13:4, 14:3, 15:2, 16:8, 17:13, 18:9, 19:17, 20:14, 21:10}
    else:print('Undefined order, use PRISM30 or PRISM63 instead.')
    return np.array([array[relabel[_]-1] for _ in relabel])

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
plt.savefig(figure_dir / 'hist_chn_layer.png', dpi=300, bbox_inches='tight')
plt.close()

n_rows = 2
n_cols = 2 + GLAYER
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols + 2, 10))
ax[1,0].scatter(intensity['CD_1_blur'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[1,0].set_ylim([-0.2,1.4])
ax[0,1].scatter(intensity['CD_2_blur'], intensity['G/A'], s=s, alpha=alpha, linewidths=None)
ax[0,1].set_ylim([-0.2,1.4])
ax[0,0].scatter(intensity['CD_1_blur'], intensity['CD_2_blur'], s=s, alpha=alpha, linewidths=None)
ax[0,0].set_xlim(XRANGE)
ax[0,0].set_ylim(YRANGE)

for subextrema in minima: ax[1,0].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')
for subextrema in minima: ax[0,1].axhline(y=subextrema, color='r', alpha=0.5, linestyle='--')

for layer in range(GLAYER):
    ax_scatter = ax[0, 2+layer]
    ax_hist = ax[1, 2+layer]
    sub = intensity[intensity['G_layer']==layer]
    ax_scatter.set_title(f'G={layer}')
    ax_scatter.scatter(sub['CD_1_blur'], sub['CD_2_blur'], s=s, alpha=alpha, linewidths=None)
    ax_scatter.set_xlim(XRANGE)
    ax_scatter.set_ylim(YRANGE)

    x, y = sub['CD_1_blur'], sub['CD_2_blur']
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    percentile = np.percentile(hist, percentile_thre)
    ax_hist.hist2d(x, y, bins=bins, vmax=percentile,
                   range=[XRANGE, YRANGE], cmap='inferno')
    ax_hist.set_xlim(XRANGE)
    ax_hist.set_ylim(YRANGE)
plt.savefig(figure_dir / 'ColorSpace_3view.png', dpi=300, bbox_inches='tight')
plt.close()

# save params
params = {
    # basic
    "PRISM_PANEL": PRISM_PANEL,
    "GLAYER": GLAYER,
    "COLOR_GRADE": COLOR_GRADE,
    "Q_CHNS": Q_CHNS,
    "Q_NUM": Q_NUM,  # 直接计算结果: 15
    "thre_min": thre_min,
    "thre_max": thre_max,

    # visualization
    "XRANGE": XRANGE,
    "YRANGE": YRANGE,
    "s": s,
    "alpha": alpha,
    "percentile_thre": percentile_thre,
    "bins": bins,  # 元组转为列表
    
    # gmm
    "CD_1_proj": CD_1_proj.tolist(),  # 转为嵌套列表
    "CD_2_proj": CD_2_proj.tolist(),
    "centroid_init_dict": centroid_init_dict,
    "colormap": colormap,
    }
with open(read_dir/'params.yaml', 'w') as f:
    yaml.dump(params, f, default_flow_style=False)

# save preprocessed data
intensity.to_csv(read_dir / 'intensity_preprocessed.csv')
intensity_G.to_csv(read_dir / 'intensity_G.csv')