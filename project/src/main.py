import datatable as dt
from sklearn.model_selection import train_test_split
from steps.ingest_data import ingest_data
from steps.preprocessing import initial_preprocessing
from steps.pre_extraction import extract_features
from steps.create_graph import create_graph
from steps.graph_operations import merge_trans_with_gf
from steps.dask_handling import create_dask_dataframe
from steps.add_edges_to_graph import add_edges_to_graph
from steps.data_split import data_split
from steps.feature_Extraction import process_graph_data
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
    train_dt, test_dt = data_split(raw_data)
    initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(train_dt, first_timestamp=-1)
    global G
    G, train_graph_ddf = create_graph(initial_preprocessed_ddf)
    graph_features = process_graph_data(train_graph_ddf, G)
    graph_features_ddf = create_dask_dataframe(graph_features)
    preprocessed_train_df = merge_trans_with_gf(train_graph_ddf, graph_features_ddf)
    
    

if __name__ == "__main__":
    main()