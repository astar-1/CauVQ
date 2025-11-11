# Path: E:\code\CausalODD\conv.py

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.inits import reset
# OGB BondEncoder is not suitable for MUTAG, so we remove it.
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter_add, scatter_min, scatter_max, scatter_mean
from torch_geometric.utils import softmax
from torch_geometric.nn.norm import GraphNorm
import math

nn_act = torch.nn.ReLU()
F_act = F.relu


# Define a new Edge Encoder for continuous edge features
class EdgeEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        # An MLP to process continuous edge features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            nn_act,
            torch.nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, edge_attr):
        # The input is already checked for None, so we can safely call float()
        return self.mlp(edge_attr.float())


class GinConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim):
        '''
            emb_dim (int): node embedding dimensionality
            edge_dim (int): edge feature dimensionality
        '''
        super(GinConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), nn_act,
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # Replace BondEncoder with our new generic EdgeEncoder
        self.edge_encoder = EdgeEncoder(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # MODIFICATION: Check if edge_attr exists before processing.
        if edge_attr is not None:
            # Process continuous edge attributes with the MLP-based encoder
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            # If no edge attributes, create a zero tensor of the correct shape.
            # The shape is (num_edges, emb_dim). emb_dim can be inferred from the mlp input.
            edge_embedding = torch.zeros(edge_index.size(1), self.mlp[0].in_features, device=x.device)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GcnConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim):
        super(GcnConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        # Replace BondEncoder with our new generic EdgeEncoder
        self.edge_encoder = EdgeEncoder(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        # MODIFICATION: Check if edge_attr exists before processing.
        if edge_attr is not None:
            # Process continuous edge attributes
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            # If no edge attributes, create a zero tensor of the correct shape.
            # The shape is (num_edges, emb_dim). emb_dim can be inferred from the linear layer output.
            edge_embedding = torch.zeros(edge_index.size(1), self.linear.out_features, device=x.device)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F_act(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class MoGnnNode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, edge_dim, drop_ratio=0.5, JK="last", residual=True, gnn_name='gin',
                 atom_encode=True):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            edge_dim (int): edge feature dimensionality
        '''
        super(MoGnnNode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.atom_encode = atom_encode

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                self.convs.append(GinConv(emb_dim, edge_dim))
            elif gnn_name == 'gcn':
                self.convs.append(GcnConv(emb_dim, edge_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.atom_encode:
            # Note: For TUDatasets like PROTEINS, node features are not categorical indices.
            # The original `atom_encoder` is designed for molecular data (e.g., OGB datasets)
            # and expects integer inputs. Here, we assume `x` is already a feature vector,
            # so we should not use atom_encoder if atom_encode is False.
            # In your `mo_model`, `atom_encode` is set to False in the `mo_TopKPoolModel` call,
            # which is correct. The node features are handled by `node_encoder` in `mo_model`.
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation