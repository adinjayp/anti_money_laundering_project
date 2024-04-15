import logging
import networkx as nx

# Configure logging
logging.basicConfig(filename='feature_extraction.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def extract_features(G, node):
    logging.info(f"Extracting features for node {node}")

    features = {}
    try:
        # Your feature extraction functions here
        # Node
        features['Node'] = int(node)
        # Degree
        features['degree'] = int(G.degree[node])
        # In Degree
        features['in_degree'] = int(G.in_degree[node])
        # Out Degree
        features['out_degree'] = int(G.out_degree[node])
        # Clustering Coefficient
        features['clustering_coefficient'] = float(nx.clustering(G, node))
        # Degree Centrality
        features['degree_centrality'] = float(nx.degree_centrality(G)[node])

        #logging.info(f"Features extracted for node {node}: {features}")
        return features

    except Exception as e:
        logging.error(f"An error occurred during feature extraction for node {node}: {e}")
        return None
