import dask.dataframe as dd

def merge_trans_with_gf(transactions_ddf, graph_features_ddf):
    
    # Merge on From_ID
    merged_ddf = dd.merge(transactions_ddf, graph_features_ddf, left_on='From_ID', right_on='Node', how='left')

    # Rename columns to avoid conflicts
    merged_ddf = merged_ddf.rename(columns={
        'degree': 'from_degree',
        'in_degree': 'from_in_degree',
        'out_degree': 'from_out_degree',
        'clustering_coefficient': 'from_clustering_coeff',
        'degree_centrality': 'from_degree_centrality'
    })
    
    # Merge on To_ID
    merged_ddf = dd.merge(merged_ddf, graph_features_ddf, left_on='To_ID', right_on='Node', how='left')

    # Rename columns again
    merged_ddf = merged_ddf.rename(columns={
        'degree': 'to_degree',
        'in_degree': 'to_in_degree',
        'out_degree': 'to_out_degree',
        'clustering_coefficient': 'to_clustering_coeff',
        'degree_centrality': 'to_degree_centrality'
    })
    
    # Drop redundant columns
    merged_ddf = merged_ddf.drop(columns=['Node_x', 'Node_y'])
    
    return merged_ddf
