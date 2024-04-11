import logging
import dask.dataframe as dd
import pandas as pd
from pre_extraction import extract_features
import pickle
import networkx as nx
import ast

# Configure logging
logging.basicConfig(filename='process_graph_data.log', level=logging.INFO)

# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def extract_graph_features(**kwargs):
    logging.info("Starting graph data processing")

    try:
        train_graph_ddf = kwargs['task_instance'].xcom_pull(task_ids='create_graph', key='G_data')['ddf']
        train_graph_ddf = pickle.loads(train_graph_ddf)
        G_bytes = kwargs['task_instance'].xcom_pull(task_ids='create_graph', key='G_data')['G']
        G = pickle.loads(G_bytes)
        # Step 1: Extract unique nodes
        unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))
        logging.info("Unique nodes extracted")

        # Step 2: Convert to Dask DataFrame
        unique_nodes_dd = dd.from_pandas(pd.DataFrame(unique_nodes, columns=['Node']), npartitions=1)
        logging.info("Unique nodes converted to Dask DataFrame")
        logging.info("Unique nodes: %s", str(unique_nodes_dd))

        # Step 3: Apply extract_features function to each partition
        graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: {key: str(value) for key, value in extract_features(G, row['Node']).items()}, axis=1))
        #logging.info("Graph features: %s", str(graph_features.compute()))
        logging.info("Graph features calculated")

        # Convert each string to a dictionary
        dicts = [ast.literal_eval(string_data) for string_data in graph_features]

        # Create a list of lists containing the dictionary values for each entry
        list_of_lists = [list(data_dict.values()) for data_dict in dicts]

        # Create a DataFrame from the list of lists
        lists_df = pd.DataFrame(list_of_lists, columns=dicts[0].keys())
        logging.info(" lists_df: %s", str(lists_df))

        # Convert specific columns to the desired data types
        convert_dtype = {'Node': 'int64', 'degree': 'int64', 'in_degree': 'int64', 'out_degree': 'int64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'}
        graph_features_df = lists_df.astype(convert_dtype)
        logging.info(" graph_features_df: %s", str(graph_features_df))
        graph_features_ddf = dd.from_pandas(graph_features_df, npartitions=1)
        
        logging.info("Dask DataFrame creation finished")
        kwargs['task_instance'].xcom_push(key='graph_features_ddf', value=graph_features_ddf)
        return graph_features_ddf

        #graph_features_bytes = pickle.dumps(graph_features)

        #kwargs['task_instance'].xcom_push(key='graph_features_bytes', value=graph_features_bytes)
        #return graph_features

    except Exception as e:
        logging.error(f"An error occurred during graph data processing: {e}")
        return None
