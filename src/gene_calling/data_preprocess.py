import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def overview(
    intensity,
    sample=10000,
    bins=50,
    ax=None,
    save=False,
    save_quality="low",
    out_path="./",
):
    data = intensity.copy()
    if sample:
        data = data.sample(sample)
    if ax is not None:
        ax_tmp = ax
    else:
        fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 5))
        ax_tmp = ax
    ax_tmp[0].hist(bins=bins, x=data["Ye/A"])
    ax_tmp[1].hist(bins=bins, x=data["B/A"])
    ax_tmp[2].hist(bins=bins, x=data["R/A"])
    ax_tmp[3].hist(bins=bins, x=data["G/A"])
    plt.tight_layout()

    if out_path.endswith(".pdf"):
        save_quality = "low"
    if save and save_quality == "low":
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    elif save and save_quality == "high":
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


RYB_x_transform = np.array([[-np.sqrt(2) / 2], [0], [np.sqrt(2) / 2]])
RYB_y_transform = np.array([[-np.sqrt(3) / 3], [2 / np.sqrt(3)], [-np.sqrt(3) / 3]])
RYB_xy_transform = np.concatenate([RYB_x_transform, RYB_y_transform], axis=1)


def gaussian_blur(
    intensity,
    RYB_x_transform=RYB_x_transform,
    RYB_y_transform=RYB_y_transform,
    channel=["Ye/A", "B/A", "R/A"],
    ref_channel=["G/A"],
    gau_0=0.03,
    gau_1=0.05,
    gau_ref=0.03,
):
    intensity_fra = intensity.copy()
    intensity_fra[ref_channel] = np.log(1 + np.log(1 + intensity_fra[ref_channel]))
    intensity_fra[ref_channel] = intensity_fra[ref_channel] / np.percentile(intensity_fra[ref_channel], 95)

    gaussian = np.concatenate([
        np.random.normal(loc=0, scale=gau_0, size=intensity_fra[channel].shape),
        np.random.normal(loc=0, scale=gau_ref, size=intensity_fra[["G/A"]].shape)], axis=1)
    intensity_fra[channel + ["G/A"]] = intensity_fra[channel + ["G/A"]].mask(intensity_fra[channel + ["G/A"]] == 0, gaussian)

    gaussian = np.random.normal(loc=0, scale=gau_1, size=intensity_fra[channel].shape)
    intensity_fra[channel] = intensity_fra[channel].mask(intensity_fra[channel] == 1, 1 + gaussian)

    intensity_fra["X_coor_gaussian"] = intensity_fra[channel].dot(RYB_x_transform)
    intensity_fra["Y_coor_gaussian"] = intensity_fra[channel].dot(RYB_y_transform)

    return intensity_fra


from scipy.signal import argrelextrema
import seaborn as sns


def plot_hist_with_extrema(a, ax=None, bins=100, extrema="max", kde_kws={"bw_adjust": 0.5}):
    sns.histplot(a, bins=bins, stat="count",
        edgecolor="white", alpha=1, ax=ax, kde=True, kde_kws=kde_kws)
    y = ax.get_lines()[0].get_ydata()
    if extrema == "max": y = -y
    extrema = [float(_ / len(y) * (max(a) - min(a)) + min(a)) for _ in argrelextrema(np.array(y), np.less)[0]]
    ax.set_title(f"{a.name}_extrema_num={len(extrema)}")
    for subextrema in extrema: ax.axvline(x=subextrema, color="r", alpha=0.5, linestyle="--")
    return extrema


def gau_hist(intensity_fra, G_layer=2, color_grade=5,
    Y_kde=0.8, B_kde=0.6, R_kde=0.6, G_kde=1, out_path="./gau_hist.png"):
    data = intensity_fra.copy()
    data = data.sample(10000)

    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(7, 7))
    plt.setp(ax, xlim=(-0.25, 1.2))
    Y_maxima = plot_hist_with_extrema(data["Ye/A"], ax=ax[0], extrema="max", kde_kws={"bw_adjust": Y_kde})
    B_maxima = plot_hist_with_extrema(data["B/A"], ax=ax[1], extrema="max", kde_kws={"bw_adjust": B_kde})
    R_maxima = plot_hist_with_extrema(data["R/A"], ax=ax[2], extrema="max", kde_kws={"bw_adjust": R_kde})
    G_minima = plot_hist_with_extrema(data["G/A"], ax=ax[3], extrema="min", kde_kws={"bw_adjust": G_kde})
    if len(R_maxima) != color_grade: R_maxima = [_ / (color_grade - 1) for _ in range(color_grade)]
    if len(Y_maxima) != color_grade: Y_maxima = [_ / (color_grade - 1) for _ in range(color_grade)]
    if len(B_maxima) != color_grade: B_maxima = [_ / (color_grade - 1) for _ in range(color_grade)]

    minima = G_minima.copy()
    minima = minima[: G_layer - 1]
    minima.insert(0, intensity_fra["G/A"].min() - 0.01)
    minima.append(intensity_fra["G/A"].max() + 0.01)

    intensity_fra["G_layer"] = pd.cut(intensity_fra["G/A"],
        bins=minima, labels=[_ for _ in range(len(minima) - 1)],
        include_lowest=True, right=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig=fig)

    ## rough division of layer
    minima = G_minima.copy()
    minima = minima[: G_layer - 1]
    minima.insert(0, intensity_fra["G/A"].min() - 0.01)
    minima.append(intensity_fra["G/A"].max() + 0.01)

    intensity_fra["G_layer"] = pd.cut(intensity_fra["G/A"], bins=minima,
        labels=[_ for _ in range(len(minima) - 1)],
        include_lowest=True, right=False)

    return Y_maxima, B_maxima, R_maxima, G_minima, intensity_fra


# preparation for init centroids
import itertools


def gau_hist_by_layer(
    intensity_fra, G_layer, color_grade,
    R_maxima, Y_maxima, B_maxima,
    Y_kde=0.8, B_kde=0.6, R_kde=0.6,
    out_path="./gau_hist_by_layer.png",
):
    centroid_init_dict = dict()
    fig, ax = plt.subplots(nrows=3, ncols=G_layer)
    for layer in range(G_layer):
        data = intensity_fra[intensity_fra["G_layer"] == layer]
        data = data.sample(10000)
        ax_tmp = ax if G_layer < 2 else ax[:, layer]
        ax_tmp[0].set_title(f"G_layer{layer}")
        Y_maxima_tmp = plot_hist_with_extrema(data["Ye/A"], ax=ax_tmp[0], extrema="max", kde_kws={"bw_adjust": Y_kde})
        B_maxima_tmp = plot_hist_with_extrema(data["B/A"], ax=ax_tmp[1], extrema="max", kde_kws={"bw_adjust": B_kde})
        R_maxima_tmp = plot_hist_with_extrema(data["R/A"], ax=ax_tmp[2], extrema="max", kde_kws={"bw_adjust": R_kde})

        if len(R_maxima_tmp) != color_grade: R_maxima_tmp = R_maxima
        if len(Y_maxima_tmp) != color_grade: Y_maxima_tmp = Y_maxima
        if len(B_maxima_tmp) != color_grade: B_maxima_tmp = B_maxima

        combinations = itertools.product(range(0, color_grade), repeat=3)
        filtered_combinations = filter(lambda x: sum(x) == color_grade - 1, combinations)
        centroid_init_dict[layer] = np.array([[Y_maxima_tmp[_[0]], B_maxima_tmp[_[1]], R_maxima_tmp[_[2]]] for _ in filtered_combinations])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig=fig)
    return centroid_init_dict



from scipy.spatial import KDTree

def remove_duplicates(coordinates, threshold=2):
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(threshold)
    print(f'{len(pairs)} duplicates pairs')
    neighbors = {}  # dictionary of neighbors
    for i, j in pairs:  # iterate over all pairs
        if i not in neighbors: neighbors[i] = set([j])
        else: neighbors[i].add(j)
        if j not in neighbors: neighbors[j] = set([i])
        else: neighbors[j].add(i)

    print(f'{len(neighbors)} neighbor entries')
    keep = []
    discard = set()  # a list would work, but I use a set for fast member testing with `in`
    nodes = set([s[0] for s in pairs]+[s[1] for s in pairs])
    for node in nodes:
        if node not in discard:  # if node already in discard set: skip
            keep.append(node)  # add node to keep list
            discard.update(neighbors.get(node, set())) # add node's neighbors to discard set
    print(f'{len(discard)} nodes discarded, {len(keep)} nodes kept')
    
    # Get all indices and subtract discard indices
    all_indices = set(range(len(coordinates)))  # All indices of coordinates
    kept_idx = list(all_indices - discard)  # Subtract discard from all indices
    centroids_simplified = coordinates[kept_idx]
    print(f'{centroids_simplified.shape[0]} centroids after simplification')
    return kept_idx, centroids_simplified


def deduplicate_df(df, columns=['Y','X'], sort_by='sum', threshold=2):
    print(f'{len(df)} raw spots.')
    df = df.sort_values(by=sort_by, ascending=False, inplace=False)
    coordinates = df[columns].values
    kept_idx, _ = remove_duplicates(coordinates, threshold)
    df_reduced = df.iloc[kept_idx]
    print(f'{len(kept_idx)} kept spots.')
    return df_reduced