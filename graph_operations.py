def merge_trans_with_gf(transactions_ddf, graph_ddf):
    # Your functions to merge transactions with graph features here
    # Create a dictionary from graph_ddf for faster lookups
    graph_dict = dict(zip(graph_ddf['Node'], graph_ddf[['degree', 'in_degree', 'out_degree', 'clustering_coefficient', 'degree_centrality']].values))
    
    def merge_partition(partition):
        
        for index, row in partition.iterrows():
            
            from_node = row['From_ID']
            to_node = row['To_ID']
            
            if from_node in graph_dict:
                graph_row = graph_dict[from_node]
                partition.loc[index, 'from_degree'] = graph_row['degree']
                partition.loc[index, 'from_in_degree'] = graph_row['in_degree']
                partition.loc[index, 'from_out_degree'] = graph_row['out_degree']
                partition.loc[index, 'from_clustering_coeff'] = graph_row['clustering_coefficient']
                partition.loc[index, 'from_degree_centrality'] = graph_row['degree_centrality']
                
            if to_node in graph_dict:
                graph_row = graph_dict[to_node]
                partition.loc[index, 'to_degree'] = graph_row['degree']
                partition.loc[index, 'to_in_degree'] = graph_row['in_degree']
                partition.loc[index, 'to_out_degree'] = graph_row['out_degree']
                partition.loc[index, 'to_clustering_coeff'] = graph_row['clustering_coefficient']
                partition.loc[index, 'to_degree_centrality'] = graph_row['degree_centrality']
                
        return partition
    
    # Apply the function to each partition
    merged_ddf = transactions_ddf.map_partitions(merge_partition)
    
    return merged_ddf
