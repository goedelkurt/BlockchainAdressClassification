import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        # input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='max'))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='max'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='max'))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
