import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))


class GCN(nn.Module):

    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward():
        pass


class MLP(nn.Module):

    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
