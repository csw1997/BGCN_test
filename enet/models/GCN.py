import torch
import torch.nn as nn
from torch.nn import functional as F
from enet.configuration import consts

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Single Layer GraphConvolution

        :param in_features: The number of incoming features
        :param out_features: The number of output features
        :param edge_types: The number of edge types in the whole graph
        :param dropout: Dropout keep rate, if not bigger than 0, 0 or None, default 0.5
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_types = consts.edge_types
        # parameters for gates
        self.Gates = nn.ModuleList()
        # parameters for graph convolutions
        self.GraphConv = nn.ModuleList()
        for _ in range(self.edge_types):
            self.Gates.append(nn.Linear(in_features=in_features, out_features=1))
            self.GraphConv.append(nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, input, adj):
        """
        :param input: FloatTensor, input feature tensor, (batch_size, seq_len, hidden_size)
        :param adj: FloatTensor (sparse.FloatTensor.to_dense()), adjacent matrix for provided graph of padded sequences, (batch_size, edge_types, seq_len, seq_len)
        :return: output
            - **output**: FloatTensor, output feature tensor with the same size of input, (batch_size, seq_len, hidden_size)
        """
        adj_ = adj.transpose(0, 1)  # (edge_types, batch_size, seq_len, seq_len)
        ts = []
        for i in range(self.edge_types):
            gate_status = F.sigmoid(self.Gates[i](input))  # (batch_size, seq_len, 1)
            adj_hat_i = adj_[i].float() * gate_status.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
            ts.append(torch.bmm(adj_hat_i, self.GraphConv[i](input)))
        ts = torch.stack(ts).sum(dim=0, keepdim=False)
        ts = F.relu(ts)
        return ts