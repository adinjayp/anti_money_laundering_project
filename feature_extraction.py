import networkx as nx

def extract_features(G, node):
    # Your feature extraction functions here
    features = {}
    #Node
    features['Node'] = node
    # Degree
    features['degree'] = G.degree[node]
    # In Degree
    features['in_degree'] = G.in_degree[node]
    # Out Degree
    features['out_degree'] = G.out_degree[node]
    # Clustering Coefficient
    features['clustering_coefficient'] = nx.clustering(G, node)
    # Degree Centrality
    features['degree_centrality'] = nx.degree_centrality(G)[node]
    
    return features
