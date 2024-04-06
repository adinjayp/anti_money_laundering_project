import networkx as nx
import pandas as pd
import dask.dataframe as dd
import datatable as dt
import numpy as np
from datetime import datetime
from datatable import f, join, sort
import pandas as pd
import dask.dataframe as dd
import sys
import os
import logging
from google.cloud import storage


# Configure logging
logging.basicConfig(filename='add_edges.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def add_edges_to_graph(ddf, G = None):
    logging.info("Starting adding edges to the graph")
    if G is None: 
        # GET G FROM BUCKET
        # Initialize a Google Cloud Storage client
        storage_client = storage.Client()

        # Specify the name of the file containing the serialized graph
        file_name = 'graph.gpickle'

        # Download the serialized graph from the bucket
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        graph_bytes = blob.download_as_string()

        # Deserialize the graph using pickle
        G = pickle.loads(graph_bytes)

        logging.info("Successfully downloaded and deserialized graph from bucket.")

    try:
        # Your functions to add edges to the graph here
        logging.debug(f"G before add edge: {G}")

        def add_edges(partition):
            G_partition = nx.DiGraph()
            for index, row in partition.iterrows():
                logging.debug(f"Adding edge from {row['From_ID']} to {row['To_ID']}")
                G_partition.add_edge(row['From_ID'], row['To_ID'], 
                                    timestamp=row['Timestamp'], 
                                    amount_sent=row['Amount_Paid'], 
                                    amount_received=row['Amount_Received'], 
                                    received_currency=row['Receiving_Currency'], 
                                    payment_format=row['Payment_Format'])
            return G_partition

        logging.debug("Computing graphs for partitions")
        graphs = ddf.map_partitions(add_edges).compute()
        logging.debug(f"Graphs partitions: {graphs}")

        logging.debug("Composing graphs")
        composed_G = nx.compose_all(graphs)
        logging.debug(f"Composed_G before merging with G: {composed_G}")

        composed_G = nx.compose_all([G] + [composed_G])
        logging.debug(f"Composed_G after merging with G: {composed_G}")

        logging.info("Finished adding edges to the graph")
        return {'G': composed_G, 'ddf': ddf}

    except Exception as e:
        logging.error(f"An error occurred during adding edges to the graph: {e}")
        return None, None
