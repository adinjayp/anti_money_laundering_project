import logging
import dask.dataframe as dd
import pandas as pd
from pre_extraction import extract_features

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

def process_graph_data(G, train_graph_ddf):
    logging.info("Starting graph data processing")

    try:
        # Step 1: Extract unique nodes
        unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))
        unique_nodes_df = pd.DataFrame(unique_nodes, columns=['Node'])

        logging.info("Unique nodes extracted")

        # Step 2: Convert to Dask DataFrame
        unique_nodes_dd = dd.from_pandas(unique_nodes_df, npartitions=2)

        logging.info("Unique nodes converted to Dask DataFrame")

        # Step 3: Calculate graph features
        graph_features = unique_nodes_dd.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))

        logging.info("Graph features calculated")

        return {'graph_features': graph_features}

    except Exception as e:
        logging.error(f"An error occurred during graph data processing: {e}")
        return None
