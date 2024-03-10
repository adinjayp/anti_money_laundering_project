import datatable as dt
from sklearn.model_selection import train_test_split
from preprocessing import initial_preprocessing, create_graph, add_edges_to_graph, initial_preprocessing_test
from graph_operations import merge_trans_with_gf
from feature_extraction import extract_features
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import numpy as np
import ast
import dask


def main():
    # Read data and perform initial preprocessing
    # raw_data = dt.fread("HI-Small_Trans.csv", columns=dt.str32)
    # Read data from GCS bucket in VM
    gcs_bucket_path = "gs://aml_mlops_bucket/HI-Small_Trans.csv"
    raw_data_pandas = pd.read_csv(gcs_bucket_path)
    raw_data = dt.Frame({col: raw_data_pandas[col].astype(str) for col in raw_data_pandas.columns})
    # raw_data = dt.fread(gcs_bucket_path, columns=dt.str32)
    train_df, test_df = train_test_split(raw_data.to_pandas(), test_size=0.2, random_state=42, stratify=raw_data['Is Laundering'])
    train_dt = dt.Frame(train_df)
    test_dt = dt.Frame(test_df)
    initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(train_dt, first_timestamp=-1)

    # Create graph
    global G
    G, train_graph_ddf = create_graph(initial_preprocessed_ddf)

    # Convert the list of unique nodes to a Dask DataFrame
    unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))

    #append to unique nodes whenever new accounts from test set come up
    unique_nodes_dd = dd.from_pandas(pd.DataFrame(unique_nodes, columns=['Node']), npartitions=2)

    # Apply extract_features function to each partition
    graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))

    # Example of delayed graph features
    graph_features = [dask.delayed(lambda x: x)(string_data) for string_data in graph_features]

    # Compute delayed objects
    graph_features_computed = dask.compute(*graph_features)

    # Convert each string to a dictionary
    dicts = [ast.literal_eval(string_data) for string_data in graph_features_computed]

    # Create a list of lists containing the dictionary values for each entry
    list_of_lists = [list(data_dict.values()) for data_dict in dicts]

    # Create a DataFrame from the list of lists
    lists_df = pd.DataFrame(list_of_lists, columns=dicts[0].keys())

    # Convert specific columns to the desired data types
    convert_dtype = {'Node': 'int64', 'degree': 'int64', 'in_degree': 'int64', 'out_degree': 'int64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'}
    graph_features_df = lists_df.astype(convert_dtype)
    graph_features_ddf = dd.from_pandas(graph_features_df, npartitions=2)


    # Merge transactions with graph features
    preprocessed_train_df = merge_trans_with_gf(train_graph_ddf, graph_features_ddf)

    # Test set prep

    test_initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing_test(test_dt, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict)
    test_graph_ddf = add_edges_to_graph(G, test_initial_preprocessed_ddf)
    unique_nodes_test = list(set(test_graph_ddf['From_ID']).union(test_graph_ddf['To_ID']))

    #append unique nodes whenever new accounts from test set come up
    unique_nodes_dd_test = dd.from_pandas(pd.DataFrame(unique_nodes_test, columns=['Node']), npartitions=2)

    graph_features_test = unique_nodes_dd_test.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))

    # Example of delayed graph features
    graph_features_test = [dask.delayed(lambda x: x)(string_data) for string_data in graph_features_test]

    # Compute delayed objects
    graph_features_computed_test = dask.compute(*graph_features_test)

    # Convert each string to a dictionary
    dicts_test = [ast.literal_eval(string_data) for string_data in graph_features_computed_test]

    # Create a list of lists containing the dictionary values for each entry
    list_of_lists_test = [list(data_dict.values()) for data_dict in dicts_test]

    # Create a DataFrame from the list of lists
    lists_df_test = pd.DataFrame(list_of_lists_test, columns=dicts[0].keys())

    # Convert specific columns to the desired data types
    convert_dtype = {'Node': 'int64', 'degree': 'int64', 'in_degree': 'int64', 'out_degree': 'int64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'}
    graph_features_df_test = lists_df_test.astype(convert_dtype)
    graph_features_ddf_test = dd.from_pandas(graph_features_df_test, npartitions=2)

    preprocessed_test_df = merge_trans_with_gf(test_graph_ddf, graph_features_ddf_test)


if __name__ == "__main__":
    main()
