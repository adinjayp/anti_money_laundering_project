import logging
import dask.dataframe as dd
import pandas as pd
from pre_extraction import extract_features
import pickle

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

def apply_extract_features(G, row):
    return extract_features(G, row['Node'])

def process_graph_data(**kwargs):
    logging.info("Starting graph data processing")

    try:
        train_graph_ddf = kwargs['task_instance'].xcom_pull(task_ids='create_graph', key='G_data')['ddf']
        train_graph_ddf = pickle.loads(train_graph_ddf)
        G_bytes = kwargs['task_instance'].xcom_pull(task_ids='create_graph', key='G_data')['G']
        G = pickle.loads(G_bytes)
        # Step 1: Extract unique nodes
        unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))
        unique_nodes_df = pd.DataFrame(unique_nodes, columns=['Node'])

        logging.info("Unique nodes extracted")

        # Step 2: Convert to Dask DataFrame
        unique_nodes_dd = dd.from_pandas(unique_nodes_df, npartitions=2)

        logging.info("Unique nodes converted to Dask DataFrame")

        # Define metadata as a Pandas DataFrame
        metadata = pd.DataFrame({
            'Node': [0],  # Sample value for the 'Node' column
            'degree': [0],  # Sample value for the 'degree' column
            'in_degree': [0],  # Sample value for the 'in_degree' column
            'out_degree': [0],  # Sample value for the 'out_degree' column
            'clustering_coefficient': [0.0],  # Sample value for the 'clustering_coefficient' column
            'degree_centrality': [0.0]  # Sample value for the 'degree_centrality' column
        })

        # Step 3: Calculate graph features
        graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(apply_extract_features, args=(G,), axis=1), meta=metadata)
        #graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))

        logging.info("Graph features calculated")
        logging.info("graph_features type: %s", str(type(graph_features)))
        logging.info("graph_features head: %s", str(graph_features.head(1)))

        kwargs['task_instance'].xcom_push(key='graph_features', value=graph_features)
        return graph_features

    except Exception as e:
        logging.error(f"An error occurred during graph data processing: {e}")
        return None
