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
from dask.distributed import Client
from google.cloud import storage
import pickle
import json
from datetime import datetime

# Address of the Dask scheduler
scheduler_address = 'tcp://10.128.0.5:8786'

# Connect to the Dask cluster
client = Client(scheduler_address)
client.upload_file('feature_extraction.py')
client.upload_file('graph_operations.py')
client.upload_file('preprocessing.py')


def main():
    # Read data and perform initial preprocessing
    # raw_data = dt.fread("HI-Small_Trans.csv", columns=dt.str32)
    # Read data from GCS bucket in VM
    gcs_bucket_path = "gs://aml_mlops_bucket/HI_Small_Trans.csv"
    raw_data_pandas = pd.read_csv(gcs_bucket_path).astype(str)
    raw_data = dt.Frame(raw_data_pandas)
    # raw_data = dt.fread(gcs_bucket_path, columns=dt.str32)
    train_df, test_df = train_test_split(raw_data.to_pandas(), test_size=0.2, random_state=42, stratify=raw_data['Is Laundering'])
    train_dt = dt.Frame(train_df).head(10000)
    test_dt = dt.Frame(test_df).head(2000)
    print("calling inititial preprocessing for train")
    initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(train_dt, first_timestamp=-1)
    print("initital preprocessing on train is done")
    print(initial_preprocessed_ddf.compute())
    # Create graph
    global G
    G, train_graph_ddf = create_graph(initial_preprocessed_ddf)
    print(G, train_graph_ddf)
    # print(f"Graph attributes: {G.nodes}, {G.edges}")
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    # Convert the list of unique nodes to a Dask DataFrame
    unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))

    #append to unique nodes whenever new accounts from test set come up
    unique_nodes_dd = dd.from_pandas(pd.DataFrame(unique_nodes, columns=['Node']), npartitions=2)

    # Apply extract_features function to each partition
    #graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))
    graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: {key: str(value) for key, value in extract_features(G, row['Node']).items()}, axis=1))

    # Example of delayed graph features
    #graph_features = [dask.delayed(lambda x: x)(string_data) for string_data in graph_features]

    # Compute delayed objects
    #graph_features_computed = dask.compute(*graph_features)

    # Convert each string to a dictionary
    dicts = [ast.literal_eval(str(string_data)) for string_data in graph_features]

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
    print(preprocessed_train_df.head())
    # Test set prep
    print("begin test set prep")
    test_initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing_test(test_dt, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict)
    G, test_graph_ddf = add_edges_to_graph(G, test_initial_preprocessed_ddf)
    print(G)
    unique_nodes_test = list(set(test_graph_ddf['From_ID']).union(test_graph_ddf['To_ID']))

    #append unique nodes whenever new accounts from test set come up
    unique_nodes_dd_test = dd.from_pandas(pd.DataFrame(unique_nodes_test, columns=['Node']), npartitions=2)

    #graph_features_test = unique_nodes_dd_test.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))
    graph_features_test = unique_nodes_dd_test.map_partitions(lambda df: df.apply(lambda row: {key: str(value) for key, value in extract_features(G, row['Node']).items()}, axis=1))


    # Example of delayed graph features
    #graph_features_test = [dask.delayed(lambda x: x)(string_data) for string_data in graph_features_test]

    # Compute delayed objects
    #graph_features_computed_test = dask.compute(*graph_features_test)

    # Convert each string to a dictionary
    dicts_test = [ast.literal_eval(str(string_data)) for string_data in graph_features_test]

    # Create a list of lists containing the dictionary values for each entry
    list_of_lists_test = [list(data_dict.values()) for data_dict in dicts_test]

    # Create a DataFrame from the list of lists
    lists_df_test = pd.DataFrame(list_of_lists_test, columns=dicts[0].keys())

    # Convert specific columns to the desired data types
    convert_dtype = {'Node': 'int64', 'degree': 'int64', 'in_degree': 'int64', 'out_degree': 'int64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'}
    graph_features_df_test = lists_df_test.astype(convert_dtype)
    graph_features_ddf_test = dd.from_pandas(graph_features_df_test, npartitions=2)

    preprocessed_test_df = merge_trans_with_gf(test_graph_ddf, graph_features_ddf_test)
    print(preprocessed_test_df.head())

    #code to push G and other files to cloud VM

    # Initialize a Google Cloud Storage client
    storage_client = storage.Client()

    # Serialize the graph to a bytes object
    graph_bytes = pickle.dumps(G)

    # Specify the name of your GCP bucket
    bucket_name = 'aml_mlops_bucket'

    # Specify the name for the file in the bucket
    file_name = 'graph.gpickle' 

    # Upload the serialized graph to the bucket
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(graph_bytes, content_type='application/octet-stream')

    print(f'Graph saved to gs://{bucket_name}/{file_name}')

    preprocessed_train_df
    # Convert DataFrame to CSV format
    preprocessed_train_df_csv = preprocessed_train_df.to_csv(index=False)
    
    # Specify file name
    file_name = 'HI_Small_Trans_preprocessed.csv'
    
    # Define destination blob in the bucket
    blob = bucket.blob(file_name)
    
    # Upload CSV data to the blob
    blob.upload_from_string(preprocessed_train_df_csv)

    # Convert the dictionary to a JSON string
    json_account_dict = json.dumps(account_dict)
    json_currency_dict = json.dumps(currency_dict)
    json_payment_format_dict = json.dumps(payment_format_dict)
    json_bank_account_dict = json.dumps(bank_account_dict)
    json_first_timestamp = json.dumps({"first_timestamp": first_timestamp})

    # Specify the name of the file to be saved in the bucket
    file_name_account_dict = "account_dict.json"
    file_name_currency_dict = "currency_dict.json"
    file_name_payment_format_dict = "payment_format_dict.json"
    file_name_bank_account_dict = "bank_account_dict.json"
    file_name_first_timestamp = "first_timestamp.json"

    # Define the blob object
    blob_data_dict = bucket.blob(file_name_account_dict)
    blob_currency_dict = bucket.blob(file_name_currency_dict)
    blob_payment_format_dict = bucket.blob(file_name_payment_format_dict)
    blob_bank_account_dict = bucket.blob(file_name_bank_account_dict)
    blob_first_timestamp = bucket.blob(file_name_first_timestamp)


    # Upload the JSON data to the bucket
    blob_data_dict.upload_from_string(json_account_dict)
    blob_currency_dict.upload_from_string(json_currency_dict)
    blob_payment_format_dict.upload_from_string(json_payment_format_dict)
    blob_bank_account_dict.upload_from_string(json_bank_account_dict)
    blob_first_timestamp.upload_from_string(json_first_timestamp)


    print("Data uploaded successfully to GCS bucket:", bucket_name)

if __name__ == "__main__":
    main()
