import torch as th
import torch.nn as nn
import torch.nn.functional as F
from graph_sage import GraphSAGEModule


class DiffPoolModule(nn.Module):
    def __init__(self, in_feats, out_feats, out_nodes, aggregator, final=False):
        super().__init__()
        self.embed = GraphSAGEModule(in_feats, out_feats, aggregator)
        self.assign = GraphSAGEModule(in_feats, out_nodes, aggregator)
        self.final = final

    def forward(self, x, adj):
        z = self.embed(x, adj)  # Eq. (5)
        s = F.softmax(self.assign(x, adj), dim=1)  # Eq. (6)
        x = th.mm(s.t(), x)  # Eq. (3)
        adj = th.mm(s.t(), th.mm(adj, s))  # Eq. (4)
        if self.final:
            return x, adj, 0
        cost = th.norm(adj - th.mm(s, s.t()))
        return x, adj, cost
