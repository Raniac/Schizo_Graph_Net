from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, \
                     Linear as Lin, \
                     ReLU

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, \
                               GATConv, \
                               GINConv, \
                               DenseSAGEConv, \
                               global_mean_pool, \
                               global_max_pool, \
                               avg_pool_x, \
                               dense_diff_pool

from torch_geometric.utils import add_self_loops

class Net_191106(torch.nn.Module):
    def __init__(self):
        super(Net_191106, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

## Graph Convolutional Network
class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 64)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        feature_after_gmp = x # output feature after conv2

        return F.log_softmax(x, dim=1), feature_after_gmp

## Graph Attention Network
class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(4, 8)
        self.conv2 = GATConv(8, 8)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)

## Graph Isomorphism Network
## GIN achieves maximal discriminative power by using injective neighbor aggregation.
class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        nn = Seq(Lin(4, 16), ReLU(), Lin(16, 4))
        self.conv = GINConv(nn, train_eps=True)
        self.feat_after_gap = []

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, batch)
        self.feat_after_gap.extend(x)

        return F.log_softmax(x, dim=1)

## Hierarchical Graph Representation Learning with Differentiable Pooling
## https://arxiv.org/abs/1806.08804
# TODO: study on GDP inputs parameters

class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(torch.cat([x1, x2], dim=-1))

class DiffPool(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(DiffPool, self).__init__()

        # num_nodes = ceil(0.25 * dataset[0].num_nodes)
        num_nodes = 90
        num_classes = 2
        self.embed_block1 = Block(4, hidden, hidden)
        self.pool_block1 = Block(4, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(0.25 * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.lin1 = Lin((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Lin(hidden, num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for block1, block2 in zip(self.embed_blocks, self.pool_blocks):
            block1.reset_parameters()
            block2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj = data.x, data.adj

        s = self.pool_block1(x, adj, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, add_loop=True))
        xs = [x.mean(dim=1)]
        x, adj, reg, _ = dense_diff_pool(x, adj, s)

        for embed, pool in zip(self.embed_blocks, self.pool_blocks):
            s = pool(x, adj)
            x = F.relu(embed(x, adj))
            xs.append(x.mean(dim=1))
            x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
