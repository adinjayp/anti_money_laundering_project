import networkx as nx
import pandas as pd
import dask.dataframe as dd
from steps.add_edges_to_graph import add_edges_to_graph
import datatable as dt
import numpy as np
from datetime import datetime
from datatable import f, join, sort
import pandas as pd
import dask.dataframe as dd
import sys
import os
import logging

logging.basicConfig(filename='graph_creation.log', level=logging.INFO)

def create_graph(ddf):
    logging.info("Starting graph creation")

    # Your graph creation functions here
    
    try:
        G = nx.DiGraph()
        G, ddf = add_edges_to_graph(G, ddf)
        logging.info("Graph edges added successfully")
        logging.debug(f"Graph info: {G}")
        logging.info("Graph creation finished")
        logging.debug("ddf head after addedge: ", ddf)
        logging.debug(f"Graph attributes: {G.nodes}, {G.edges}")
        logging.debug("Number of nodes:", G.number_of_nodes())
        logging.debug("Number of edges:", G.number_of_edges())
        return G, ddf

    except Exception as e:
        logging.error(f"An error occurred during graph creation: {e}")
        return None, None