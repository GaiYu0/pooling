import networkx as nx
import torch as th


class DataLoader:
    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError()

    @property
    def n_feats():
        raise NotImplementedError()


class BarabasiAlbertGraphLoader(DataLoader):
    def __init__(n, m, device):
        self.n = n
        self.m = m
        self.device = device

    def next(self):
        while True:
            g = nx.barabasi_albert_graph(n, m)
            adj = th.from_numpy(nx.adjacency_matrix(g).todense()).to(self.device)
            x = th.sum(adj, dim=1, keepdim=True)
            yield x, adj

    @property
    def n_feats():
        return 1
