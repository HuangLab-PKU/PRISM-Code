import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


global num_per_layer, G_layer, cluster_num
# def calculate_cdf(
#     intensity,
#     st,
#     num_per_layer,
#     channel=["Ye/A", "B/A", "R/A"],
#     gmm=None,
# ):
#     X_sub_cal = intensity[channel]
#     # Get the of each cluster
#     cdfs_df = pd.DataFrame()
#     if gmm is not None:
#         for i in tqdm(range(gmm.n_components), desc="component"):
#             if gmm.covariance_type == "tied" or "full":
#                 mean = gmm.means_[i]
#                 cov = gmm.covariances_
#             elif gmm.covariance_type == "diag" or "spherical":
#                 mean = gmm.means_[i]
#                 cov = np.diag(gmm.covariances_[i])

#             m_dist_x = (X_sub_cal - mean) @ np.linalg.inv(cov)
#             m_dist_x = np.einsum("ij,ji->i", m_dist_x, (X_sub_cal - mean).T)

#             probability = 1 - stats.chi2.cdf(np.array(m_dist_x), 3)
#             cdfs_df[i + 1 + st] = probability

#     else:
#         # Get the of each cluster
#         cdfs_df = pd.DataFrame()
#         # centroids = np.array([0,0,0])

#         for i in tqdm(range(st + 1, st + num_per_layer + 1), desc="component"):
#             data_cdf = intensity[channel]

#             data = intensity[intensity["label"] == i]
#             data = data[channel]
#             # print(data)
#             # Example points (replace this with your actual data)
#             # print(data)
#             points = np.array(data)

#             # Calculate the mean
#             mean = np.mean(points, axis=0)
#             # centroids = np.concatenate([centroids, [mean]], axis=1)

#             # Calculate the covariance matrix
#             cov = np.cov(points, rowvar=False)
#             # cov = np.diag(cov)
#             # print(f'mean: {mean}\ncov:\n{cov}')
#             m_dist_x = (data_cdf - mean) @ np.linalg.pinv(cov)
#             m_dist_x = np.einsum("ij,ji->i", m_dist_x, (data_cdf - mean).T)

#             probability = 1 - stats.chi2.cdf(np.array(m_dist_x), 3)
#             cdfs_df[i] = probability
#         # print(centroids)
#         # centroids = centroids[1:,:]

#     cdfs_df.index = intensity.index

#     return cdfs_df

def calculate_cdf(intensity, st, num_per_layer, channel=['Ye/A', 'B/A', 'R/A',]):
    centroids = []
    cdfs_df = pd.DataFrame()
    for i in tqdm(range(st + 1, st + num_per_layer + 1), desc='component'):
        data_cdf = intensity[channel]
        data = intensity[intensity['label'] == i]
        data = data[channel]
        points = np.array(data)
        
        # calculate the mean
        mean = np.mean(points, axis=0)

        # calculate the covariance matrix
        cov = np.cov(points, rowvar=False)

        # calculate cdf
        m_dist_x = (data_cdf - mean) @ np.linalg.pinv(cov)
        m_dist_x = np.einsum('ij,ji->i', m_dist_x, (data_cdf - mean).T)
        probability = 1 - stats.chi2.cdf(np.array(m_dist_x), len(channel))
        cdfs_df[i] = probability
        centroids.append(mean)
    centroids = np.array(centroids)
    cdfs_df.index = intensity.index

    return cdfs_df, centroids

def conut_distribution(intensity_fra, num_per_layer, G_layer, out_path):
    # Barplot
    plt.figure(figsize=(num_per_layer * G_layer / 7, 3))
    sns.barplot(
        x=[cluster_num + 1 for cluster_num in range(num_per_layer * G_layer)],
        y=[
            len(intensity_fra[intensity_fra["label"] == cluster_num + 1])
            for cluster_num in range(num_per_layer * G_layer)
        ],
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def cdf_layer(cdf_data, layer):
    cm = sns.clustermap(
        cdf_data.sample(1000),
        figsize=(5, 4),
        metric="euclidean",
        method="ward",
        col_cluster=False,
    )
    cm.ax_heatmap.set(
        yticks=[],
    )
    cm.fig.suptitle(f"layer={layer}")


def cdf_heatmap(
    intensity_fra,
    CDF_dict,
    p_thre_list=[0.0001, 0.001, 0.01, 0.1],
    corr_method="spearman",
    out_path="./cdf_heatmap.jpg",
    G_layer=2,
    num_per_layer=15,
):
    fig, ax = plt.subplots(
        nrows=G_layer,
        ncols=len(p_thre_list) + 1,
        figsize=((len(p_thre_list) + 1), G_layer),
    )
    for layer in tqdm(range(G_layer)):
        cdfs_df = CDF_dict[layer]
        X_sub = intensity_fra[intensity_fra["G_layer"] == layer]
        ax_heat = ax[layer, -1] if G_layer > 1 else ax[-1]
        corr_matrix = cdfs_df.corr(method=corr_method)
        sns.heatmap(
            corr_matrix,
            ax=ax_heat,
            cmap="coolwarm",
        )
        ax_heat.set_title(f"{corr_method}_correlation")

        for _, p_thre in enumerate(p_thre_list):
            overlap = pd.DataFrame()

            for cluster_num in range(
                layer * num_per_layer + 1, (layer + 1) * num_per_layer + 1
            ):
                tmp = cdfs_df.loc[X_sub["label"][X_sub["label"] == (cluster_num)].index]
                overlap[cluster_num] = (tmp > p_thre).sum(axis=0) / len(tmp)

            ax_tmp = (
                ax[layer, _]
                if len(p_thre_list) > 1 and G_layer > 1
                else ax[layer]
                if G_layer > 1
                else ax[_]
                if len(p_thre_list) > 1
                else ax
            )
            ax_tmp.set_title(f"p_thre = {p_thre}")
            if _ == 0:
                ax_tmp.set_ylabel(f"G={layer}", fontsize=16)

            sns.heatmap(overlap, vmin=0, vmax=1, ax=ax_tmp)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig=fig)


def thre_by_quantile(
    intensity_fra,
    CDF_dict,
    quantile_thre=0.1,
    cluster_thre=[],
    out_path="./quantile.jpg",
):
    for layer in range(G_layer):
        cdfs_df = CDF_dict[layer]
        X_sub = intensity_fra[intensity_fra["G_layer"] == layer]
        for cluster_num in range(
            layer * num_per_layer + 1, (layer + 1) * num_per_layer + 1
        ):
            tmp = cdfs_df.loc[X_sub["label"][X_sub["label"] == (cluster_num)].index]
            quantile = tmp[cluster_num].quantile(quantile_thre)
            cluster_thre.append(quantile)

    thre_index = []
    for layer in range(G_layer):
        cdfs_df = CDF_dict[layer]
        X_sub = intensity_fra[intensity_fra["G_layer"] == layer]
        for cluster_num in range(
            layer * num_per_layer + 1, (layer + 1) * num_per_layer + 1
        ):
            p_thre = cluster_thre[cluster_num - 1]
            tmp = cdfs_df.loc[X_sub["label"][X_sub["label"] == (cluster_num)].index]
            tmp = tmp[tmp[cluster_num] > p_thre]
            thre_index += list(tmp.index)

    thre_index.sort()
    thre_index = pd.Index(thre_index)
    intensity_fra_thre = intensity_fra.loc[thre_index]

    print(f"points_kept: {len(thre_index) / len(intensity_fra) * 100 :.1f}%")
    plt.figure(figsize=(num_per_layer * G_layer / 3, 5))
    sns.barplot(x=[_ + 1 for _ in range(len(cluster_thre))], y=cluster_thre)
    plt.savefig(out_path)
    return intensity_fra_thre


def plot_mean_accuracy(cdfs_df, X_sub, sample=50, y_line=0.95, total_num=15, ax=None):
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
