import networkx as nx
import torch as th


def barabasi_albert_graph_loader(n, m, device):
    while True:
        g = nx.barabasi_albert_graph(n, m)
        adj = th.from_numpy(nx.adjacency_matrix(g).todense())
        yield x, adj
