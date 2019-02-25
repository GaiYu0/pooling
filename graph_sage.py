import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEModule(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator='mean'):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
#       nn.init.xavier_uniform_(self.linear.weight)
        self.aggregator = aggregator

    def forward(self, x, adj):
        if self.aggregator == 'mean':
            deg = th.sum(adj, dim=1, keepdim=True)
            h_N = th.mm(adj, x) / deg
        h = F.relu(self.linear(th.cat([x, h_N]))
        norm = th.norm(h, dim=1, keepdim=True) + 1e-5
        h = h / norm
        return h
