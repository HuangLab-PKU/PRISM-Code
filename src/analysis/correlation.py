def matrix_for_heatmap(adata_1, adata_2, adata_ori, obs_1="leiden_type", obs_2="leiden", cluster_of_intere=list, sc_cluster_of_intere=list, save=False, whole=False):
    
    raw_cluster_num = len(cluster_of_intere)
    sc_cluster_num = len(sc_cluster_of_intere)
    raw_data_matrix = np.array([])
    raw_data_whole_matrix = np.array([])
    sc_data_matrix = np.array([])

    for cluster_num in cluster_of_intere:
        if whole:
            raw_add_whole = np.array(
                [
                    np.mean(
                        adata_ori[adata_1[adata_1.obs[obs_1] == str(cluster_num)].obs.index].X,
                        axis=0,
                    )
                ]
            )
            if raw_data_whole_matrix.size == 0:
                raw_data_whole_matrix = raw_add_whole
            else:
                raw_data_whole_matrix = np.concatenate((raw_data_whole_matrix, raw_add_whole), axis=0)

        raw_add = np.array(
            [
                np.mean(
                    adata_1.X[
                        adata_1.obs[obs_1] == str(cluster_num)
                    ],
                    axis=0,
                )
            ]
        )
        if raw_data_matrix.size == 0:
            raw_data_matrix = raw_add
        else:
            raw_data_matrix = np.concatenate((raw_data_matrix, raw_add), axis=0)


    for cluster_num in sc_cluster_of_intere:
        sc_add = np.array(
            [
                np.mean(
                    adata_2.X[
                        adata_2.obs[obs_2] == str(cluster_num)
                    ], 
                    axis=0
                )
            ]
        )
        if sc_data_matrix.size == 0:
            sc_data_matrix = sc_add
        else:
            sc_data_matrix = np.concatenate((sc_data_matrix, sc_add), axis=0)

    matrix = np.concatenate((raw_data_matrix, sc_data_matrix), axis=0)
    corr_matrix = np.corrcoef(matrix)
    if whole:
        return raw_data_whole_matrix, sc_data_matrix, corr_matrix[0 : raw_cluster_num, raw_cluster_num : raw_cluster_num + sc_cluster_num]
    else:
        return raw_data_matrix, sc_data_matrix, corr_matrix[0 : raw_cluster_num, raw_cluster_num : raw_cluster_num + sc_cluster_num]
