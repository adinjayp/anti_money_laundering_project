import networkx as nx
import pandas as pd
import dask.dataframe as dd
from add_edges_to_graph import add_edges_to_graph
import datatable as dt
import numpy as np
from datetime import datetime
from datatable import f, join, sort
import pandas as pd
import dask.dataframe as dd
import sys
import os
import logging
import pickle

logging.basicConfig(filename='graph_creation.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def create_graph(**kwargs):
    logging.info("Starting graph creation")

    # Your graph creation functions here
    
    try:
        ddf = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['ddf']
        ddf = pickle.loads(ddf)
        G = nx.DiGraph()
        G, ddf = add_edges_to_graph(ddf, G)
        logging.info("Graph edges added successfully")
        logging.debug(f"Graph info: {G}")
        logging.info("Graph creation finished")
        logging.debug("ddf head after addedge: ", ddf)
        logging.debug(f"Graph attributes: {G.nodes}, {G.edges}")
        logging.debug("Number of nodes:", G.number_of_nodes())
        logging.debug("Number of edges:", G.number_of_edges())
        
        G_bytes = pickle.loads(G)
        ddf = pickle,loads(ddf)
        G_data = {
            'G': G_bytes,
            'ddf': ddf,
        }
        # Push the dictionary to XCom
        kwargs['task_instance'].xcom_push(key='G_data', value=G_data)
        return {'G': G, 'ddf': ddf}

    except Exception as e:
        logging.error(f"An error occurred during graph creation: {e}")
        return None, None
