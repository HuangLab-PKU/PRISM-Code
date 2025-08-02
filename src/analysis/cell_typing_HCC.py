def threshold_in_cluster(adata, marker_gene=[], thre_gene=['AFP','GPC3','ACTA2','PECAM1'], type_name=[], cluster_dict={}):
# for cluster_gene in cluster_to_filter:
    thre_min = [True] * len(marker_gene) + [False] * len(thre_gene)
    gene_list = marker_gene + thre_gene
    minima_dict = {}
    for _ in gene_list:
        minima_dict[_] = ''

    cluster_list_temp=[]
    for _ in cluster_dict.keys():
        for name in type_name:
            if name in _:
                cluster_list_temp += [str(_) for _ in cluster_dict[_]]

    cluster = adata[adata.obs.leiden.isin(cluster_list_temp)]

    fig, ax = plt.subplots(nrows=1,ncols=len(thre_min),figsize=(24, 4))
    for i, gene in enumerate(gene_list):
        a = [float(_) for _ in cluster[:, gene].X]
        sns.histplot(a, bins=20, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    log=True, 
                    ax=ax[i],
                    line_kws=dict(color='black', alpha=0.7, linewidth=1.5, label='KDE'))
        ax[i].get_lines()[0].set_color('red') # edit line color due to bug in sns v 0.11.0
        ax[i].set_xlabel(gene)

        y = ax[i].get_lines()[0].get_ydata()
        minima_dict[gene] = [float(_/len(y)*(max(a)-min(a))+min(a)) for _ in argrelextrema(np.array(y), np.less)[0]]
        # print(f'{gene}_minima: {minima_dict[gene]}')
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.4)
        fig.suptitle(f'distribution of cluster, marker gene={marker_gene}')
    plt.show()

    cluster.obs['tmp_leiden'] = ['-1']*len(cluster)
    for _, gene in enumerate(gene_list):
        minima = minima_dict[gene]
        while True:
            if len(minima) == 0:
                minima = [0]
                break
            if minima[0] > 1 and gene != 'CPA3':
                minima[0] = 0
                break
            if minima[0] < -1 and gene != 'CPA3':
                minima.pop(0)
                continue
            break
        
        print(f'{gene}_thre: {minima[0]}')

        if thre_min[_]:
            tmp = cluster[cluster[:, gene].X > minima[0]]
            cluster.obs['tmp_leiden'][tmp.obs.index] = ['1']*len(tmp)
        else:
            tmp = cluster[cluster[:, gene].X > minima[0]]
            cluster.obs['tmp_leiden'][tmp.obs.index] = ['-1']*len(tmp)
    
    tmp = cluster[cluster.obs['tmp_leiden']=='-1']
    adata.obs['tmp_leiden'][tmp.obs.index] = ['-2']*len(tmp)

    cell_to_plot = len(cluster[cluster.obs['tmp_leiden']=='1'])
    print(f'marker_gene={marker_gene}, {cell_to_plot} cells of {len(cluster)} cells left\n')
    return adata


def collect_liver(combine_adata_st, tissue_obs='tissue', in_out_leiden='tmp_leiden'):
    other_cluster = combine_adata_st[combine_adata_st.obs[in_out_leiden]=='-2']
    liver = other_cluster[other_cluster.obs[tissue_obs] == "liver"]
    combine_adata_st.obs[in_out_leiden] = list(combine_adata_st.obs[in_out_leiden])
    combine_adata_st.obs[in_out_leiden][liver.obs.index] = ["-1"] * len(liver)
    return combine_adata_st
