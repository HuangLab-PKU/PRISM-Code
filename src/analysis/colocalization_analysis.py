import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 设置包路径
package_path = r'E:\TMC\PRISM_Code\analysis_cell_typing'
if package_path not in sys.path:
    sys.path.append(package_path)

# 设置目录
BASE_DIR = Path(r'F:\spatial_data\processed')
RUN_ID = '20241216_ZCH_TNBC_BZ01_CA2_5um_TCR_Tcell_only'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
seg_dir = src_dir / 'segmented'
analysis_dir = src_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

# 查看并设置递归深度
print("当前递归深度：", sys.getrecursionlimit())
sys.setrecursionlimit(10000)
print("新的递归深度：", sys.getrecursionlimit())

## 加载表达数据
df = pd.read_csv(seg_dir / "expression_matrix.csv", index_col=0)
df.drop('False_pos', axis=1, inplace=True)

# 创建 AnnData 对象
adata = sc.AnnData(df)
adata.var.index = adata.var.index.str.upper()
adata.obs['dataset'] = ["PRISM"] * len(adata)
adata.obs['tissue'] = ['TNBC_BZ01_CA2'] * len(adata)
adata.raw = adata
gene_list = adata.var.index

# 加载空间信息
centroid = pd.read_csv(seg_dir / 'dapi_predict.csv')
centroid.index = centroid.index.astype(str)
centroid_sub = centroid.loc[adata.obs.index]
adata.obsm['spatial'] = np.array([centroid_sub['Y'], centroid_sub['X']]).T

## 基因共定位
exp_mtx = pd.read_csv(seg_dir / 'expression_matrix.csv')
exp_mtx = exp_mtx.iloc[:, 1:]
exp_mtx.drop('False_pos', axis=1, inplace=True)
print(f"Expression matrix shape: {exp_mtx.shape}")
print(exp_mtx.head())

df_corr = exp_mtx.copy()
correlation_matrix = df_corr.corr()
np.fill_diagonal(correlation_matrix.values, np.nan)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Gene Expression Correlation', fontsize=16)
plt.savefig(analysis_dir / 'gene_expression_correlation.png')
plt.close()

## 详细预处理
sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# 标准化数据
sc.pp.regress_out(adata, ["total_counts"])
sc.pp.scale(adata)

# 将 adata.X 转换为 DataFrame，并设置列名
data = pd.DataFrame(adata.X, columns=adata.var_names)

# 处理缺失值和无穷值
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"数据形状: {data.shape}")

## 使用 K-Means 进行聚类
n_clusters = 10  # 根据数据特性选择合适的簇数
print(f"正在执行 K-Means 聚类，簇数：{n_clusters} ...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(data)
labels = kmeans.labels_
print("K-Means 聚类完成。")

# 将聚类标签添加到 DataFrame
data['cluster'] = labels

# 按照聚类标签排序
df_sorted = data.sort_values('cluster').drop('cluster', axis=1)

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(
    df_sorted, 
    cmap='coolwarm', 
    xticklabels=df_sorted.columns,  # 设置x轴标签为列名
    yticklabels=False, 
    vmax=2, 
    vmin=-2
)
plt.title(f'Heatmap Sorted by K-Means Clusters (k={n_clusters})', fontsize=16)
plt.savefig(analysis_dir / f'heatmap_kmeans_clusters_{n_clusters}.png')
plt.close()
print("热图已保存。")

# ## 使用层次聚类绘制 Clustermap
# import fastcluster
# from scipy.cluster.hierarchy import dendrogram

# # 定义链接方法列表
# methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

# for method in tqdm(methods, desc="Linkage Methods"):
#     try:
#         print(f"正在使用链接方法: {method} ...")
#         # 使用 fastcluster 进行层次聚类
#         Z = fastcluster.linkage(data.drop('cluster', axis=1).values, method=method)
        
#         # 绘制 clustermap 并传入预计算的链接矩阵
#         g = sns.clustermap(
#             data.drop('cluster', axis=1),
#             cmap='coolwarm',
#             yticklabels=False,
#             row_cluster=True,
#             col_cluster=False,
#             row_linkage=Z,
#             vmax=3,
#             vmin=-3,
#         )
#         plt.title(f'Clustermap with {method} linkage', fontsize=16)
#         plt.savefig(analysis_dir / f'clustermap_{method}.png')
#         plt.close()
#         print(f"{method} linkage Clustermap 已保存。")
#     except RecursionError as re:
#         print(f"链接方法 {method} 导致 RecursionError: {re}，跳过此方法。")
#     except Exception as e:
#         print(f"链接方法 {method} 运行失败: {e}，跳过此方法。")
