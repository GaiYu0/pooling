import torch as th
import torch.nn as nn
import torch.nn.functional as F
from diff_pool import DiffPoolModule


class Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.modules = nn.ModuleList([DiffPoolModule(**kwargs) for kwargs in configs])
        self.modules[-1].final = True
        self.linear = nn.Linear(configs[-1]['out_feats'], 1)

    def forward(self, x, adj):
        cost = 0
        for module in self.modules:
            x, adj, c = module(x, adj)
            cost += c
        p = th.sigmoid(x)
        return p, cost

class Generator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.modules = nn.ModuleList([DiffPoolModule(**kwargs) for kwargs in configs])
        self.modules[-1].final = True

    def forward(self, x):
        adj = th.ones(1).to(x.device)
        cost = 0
        for module in self.modules:
            x, adj, c = module(x, adj)
            cost += c
        return adj, cost
