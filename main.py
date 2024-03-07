import datatable as dt
from sklearn.model_selection import train_test_split
from preprocessing import initial_preprocessing, create_graph, add_edges_to_graph
from graph_operations import merge_trans_with_gf
from feature_extraction import extract_features
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import numpy as np


def main():
    # Read data and perform initial preprocessing
    raw_data = dt.fread("HI-Small_Trans.csv", columns=dt.str32, fill=True)
    train_dt, test_dt = train_test_split(raw_data.to_pandas(), test_size=0.2, random_state=42, stratify=raw_data['Is Laundering'])
    initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(train_dt, first_timestamp=-1)

    # Create graph
    global G
    G, train_graph_ddf = create_graph(initial_preprocessed_ddf)

    # Convert the list of unique nodes to a Dask DataFrame
    unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))

    #append to unique nodes whenever new accounts from test set come up
    unique_nodes_dd = dd.from_pandas(pd.DataFrame(unique_nodes, columns=['Node']), npartitions=2)

    # Apply extract_features function to each unique node
    graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(row['Node']), axis=1), meta={'degree': 'float64', 'in_degree': 'float64', 'out_degree': 'float64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'})

    # Persist the result in memory
    graph_features = graph_features.persist()

    # Add new columns to transactions_ddf
    train_graph_ddf['from_degree'] = None
    train_graph_ddf['from_in_degree'] = None
    train_graph_ddf['from_out_degree'] = None
    train_graph_ddf['from_clustering_coeff'] = None
    train_graph_ddf['from_degree_centrality'] = None
    train_graph_ddf['to_degree'] = None
    train_graph_ddf['to_in_degree'] = None
    train_graph_ddf['to_out_degree'] = None
    train_graph_ddf['to_clustering_coeff'] = None
    train_graph_ddf['to_degree_centrality'] = None
    

    # Merge transactions with graph features
    preprocessed_train_df = merge_trans_with_gf(train_graph_ddf)

    # Continue with your remaining code logic


if __name__ == "__main__":
    main()
