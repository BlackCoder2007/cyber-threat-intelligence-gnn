import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Basic GCN layer: H' = A_hat H W."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, adjacency):
        return adjacency @ self.linear(x)


class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_features=32, num_classes=2, dropout=0.25):
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, num_classes)
        self.dropout = dropout

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, adjacency)
