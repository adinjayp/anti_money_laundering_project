import datatable as dt
from sklearn.model_selection import train_test_split
from steps.ingest_data import ingest_data
from steps.preprocessing import initial_preprocessing
from steps.create_graph import create_graph
from steps.feature_extraction import extract_features
from steps.dask_handling import create_dask_dataframe
from steps.graph_operations import merge_trans_with_gf
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import numpy as np
import ast
import dask
from dask.distributed import Client

# Address of the Dask scheduler
scheduler_address = 'tcp://10.128.0.5:8786'

# Connect to the Dask cluster
client = Client(scheduler_address)
client.upload_file('feature_extraction.py')
client.upload_file('graph_operations.py')
client.upload_file('preprocessing.py')


def main():
    raw_data = ingest_data()
    train_df, test_df = train_test_split(raw_data.to_pandas(), test_size=0.2, random_state=42, stratify=raw_data['Is Laundering'])
    train_dt = dt.Frame(train_df)
    test_dt = dt.Frame(test_df)
    initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(train_dt, first_timestamp=-1)
    global G
    G, train_graph_ddf = create_graph(initial_preprocessed_ddf)
    unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID'])) # Convert the list of unique nodes to a Dask DataFrame
    unique_nodes_dd = dd.from_pandas(pd.DataFrame(unique_nodes, columns=['Node']), npartitions=2)
    graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))
    graph_features_ddf = create_dask_dataframe(graph_features)
    preprocessed_train_df = merge_trans_with_gf(train_graph_ddf, graph_features_ddf)
    
    

if __name__ == "__main__":
    main()