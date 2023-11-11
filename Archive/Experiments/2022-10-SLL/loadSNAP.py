import json
import pandas as pd
import networkx as nx

def load_graph(graph_path):
    """
    Reading a NetworkX graph.
    :param graph_path: Path to the edge list.
    :return graph: NetworkX object.
    """
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def load_features(features_path):
    """
    Reading the features from disk.
    :param features_path: Location of feature JSON.
    :return features: Feature hash table.
    """
    features = json.load(open(features_path))
    features = {int(k): [int(val) for val in v] for k, v in features.items()}
    return features

def load_labels(labels_path):
    data = pd.read_csv(labels_path)
    labels = data.values.tolist()
    labels = [int(edge[1]) for edge in labels]
    return labels