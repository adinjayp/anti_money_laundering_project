import logging
from dask import dataframe as dd
import pandas as pd

# Configure logging
logging.basicConfig(filename='extract_unique_nodes.log', level=logging.INFO)

def extract_unique_nodes(train_graph_ddf):
    logging.info("Starting extraction of unique nodes")

    try:
        unique_nodes = list(set(train_graph_ddf['From_ID']).union(train_graph_ddf['To_ID']))
        unique_nodes_df = pd.DataFrame(unique_nodes, columns=['Node'])

        logging.info("Extraction of unique nodes finished")
        return unique_nodes_df

    except Exception as e:
        logging.error(f"An error occurred during extraction of unique nodes: {e}")
        return None
