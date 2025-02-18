import networkx as nx
from community import community_louvain

def build_graph(cointegrated_pairs):
    """Build a graph from cointegrated pairs."""
    G = nx.Graph()
    for pair in cointegrated_pairs:
        sym_1, sym_2 = pair['pair'].split('-')
        weight = 1 / pair['eg_pvalue']  # Inverse of p-value as edge weight
        G.add_edge(sym_1, sym_2, weight=weight)
    return G

def detect_communities(G):
    """Detect communities using Louvain method."""
    partition = community_louvain.best_partition(G)
    return partition
