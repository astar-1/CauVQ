from torch_geometric.nn import global_mean_pool, TopKPooling, GATConv
import torch.nn.functional as F
import torch
from causal import DegreeLayer
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Data, Batch
from utils import min_max_normalize
from conv import GnnNode, GnnNodeVirtualnode
from GCN_Conv import GcnConv
from mo_conv import MoGnnNode


# Codebook Encoding Model
class CodebookEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):  # Suggest output_dim aligns with the main model
        super().__init__()
        # Assuming GcnConv definition is GcnConv(in_channels, out_channels)
        self.conv1 = GcnConv(input_dim, hidden_dim)
        self.conv2 = GcnConv(hidden_dim, output_dim)

    def forward(self, data):
        # Ensure data.x is the correct float type
        x = self.conv1(data.x.float(), data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        return global_mean_pool(x, data.batch)


class TopKPoolModel(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, vir, JK="last", drop_ratio=0.5, gnn_type='gin', hidden_pool_dim=None,
                 atom_encode=False):
        super().__init__()
        if vir:
            self.gnn_node = GnnNodeVirtualnode(num_layer, emb_dim, JK="last", drop_ratio=drop_ratio, residual=True,
                                               gnn_name=gnn_type, atom_encode=atom_encode)
        else:
            self.gnn_node = GnnNode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=True, gnn_name=gnn_type,
                                    atom_encode=atom_encode)

        pool_input_dim = hidden_pool_dim if hidden_pool_dim is not None else emb_dim
        self.pool1 = TopKPooling(pool_input_dim, ratio=0.3)

    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None and not data.x.is_floating_point():
            pass

        if hasattr(data, 'edge_attr') and data.edge_attr is not None and not data.edge_attr.is_floating_point():
            pass

        x = self.gnn_node(data)

        if not x.is_floating_point():
            x = x.float()

        x_pooled, edge_index_pooled, _, batch_pooled, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        return x_pooled, batch_pooled


class MoTopKPoolModel(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, edge_dim, JK="last", drop_ratio=0.5, gnn_type='gin', hidden_pool_dim=None,
                 atom_encode=False):
        super().__init__()

        self.gnn_node = MoGnnNode(num_layer, emb_dim, edge_dim=edge_dim, JK=JK, drop_ratio=drop_ratio, residual=True,
                                  gnn_name=gnn_type,
                                  atom_encode=atom_encode)

        pool_input_dim = hidden_pool_dim if hidden_pool_dim is not None else emb_dim
        self.pool1 = TopKPooling(pool_input_dim, ratio=0.3)

    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None and not data.x.is_floating_point():
            pass

        if hasattr(data, 'edge_attr') and data.edge_attr is not None and not data.edge_attr.is_floating_point():
            pass

        x = self.gnn_node(data)

        if not x.is_floating_point():
            x = x.float()

        x_pooled, edge_index_pooled, _, batch_pooled, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        return x_pooled, batch_pooled


# Causal-Enhanced Codebook (Vector Quantizer)
class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings, codebook_dim, embedding_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(codebook_dim, 128)
        self.fc2 = torch.nn.Linear(128, embedding_dim)
        # Parameter representing the causal influence matrix, initialized randomly
        self.causal = torch.nn.Parameter(torch.randn(num_embeddings, num_embeddings, device='cuda'), requires_grad=True)
        self.identity_matrix_const = torch.eye(num_embeddings, dtype=torch.float32)
        self.counter_layer = DegreeLayer()

    def forward(self, codebook_input):
        current_device = self.causal.device
        identity_matrix = self.identity_matrix_const.to(current_device)

        codebook_transformed = self.fc1(codebook_input)
        codebook_transformed = self.fc2(codebook_transformed.sigmoid())

        # Create a matrix with 0 on the diagonal and 1 elsewhere
        identity_matrix_ones_subtracted = 1.0 - identity_matrix

        # The causal effect matrix A (where A_ii=1, A_ij=Causal_ij for i!=j)
        causal_effect = self.causal * identity_matrix_ones_subtracted + identity_matrix

        codebook_sig = codebook_transformed.sigmoid()

        # Causal Codebook: Only diagonal terms of Causal matrix are used first, then multiplied by causal_effect
        causal_codebook = (self.causal * identity_matrix) @ codebook_sig
        causal_codebook = torch.matmul(causal_effect, causal_codebook)

        # Counterfactual Matrix M (learned mask based on the causal diagonal)
        counter_matrix = self.counter_layer(self.causal)

        # Counterfactual Codebook: M is applied before the causal effect matrix
        counter_codebook = torch.matmul(causal_effect, torch.matmul(counter_matrix, codebook_sig))

        # Note: causal_matrix returned here is self.causal * identity_matrix (diagonal of causal matrix)
        return codebook_transformed, causal_codebook, counter_codebook, self.causal * identity_matrix, counter_matrix


# Adjacency Matrix Reconstruction (Decoder)
class SimilarityDecoder(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, batch):
        if z is None or z.numel() == 0: return []
        if batch is None or batch.numel() == 0:
            if z.ndim == 2:
                num_graphs = 1
                graph_masks = [torch.ones(z.size(0), dtype=torch.bool, device=z.device)]
            else:
                return []
        else:
            num_graphs = batch.max().item() + 1
            graph_masks = [(batch == i) for i in range(num_graphs)]

        norm_z = F.normalize(z, p=2, dim=1)
        batch_adj_recon = []
        for graph_mask in graph_masks:
            if not graph_mask.any():
                continue
            graph_z = norm_z[graph_mask]
            adj_recon = torch.mm(graph_z, graph_z.t())
            batch_adj_recon.append(torch.sigmoid(adj_recon / self.temperature))
        return batch_adj_recon


# Classifier Layer
class ClassLayer(torch.nn.Module):
    def __init__(self, embedding_dim, num_tasks, temperature=1.0):
        super().__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_tasks)
        self.temperature = temperature

    def forward(self, x, batch, codebook, causal_codebook, counter_codebook):
        if x is None or x.numel() == 0:
            raise ValueError("Input x to ClassLayer is empty after pooling.")
        pooled_x = global_mean_pool(x, batch)

        # Find nearest neighbor in causal_codebook for each pooled node feature x
        distances_to_causal_cb = torch.cdist(x, causal_codebook) ** 2
        indices = torch.argmin(distances_to_causal_cb, dim=1)

        # Select corresponding vectors from causal and counterfactual codebooks
        selected_causal_vectors = causal_codebook[indices]
        selected_counter_vectors = counter_codebook[indices]

        # Causal and Counterfactual Node Representations
        causal_output_nodes = x + selected_causal_vectors
        counter_output_nodes = selected_counter_vectors  # Note: This only uses the counterfactual vector, not 'x + ...'

        # Select corresponding vectors from the processed codebook (z_nodes)
        z_nodes = codebook[indices]

        # Global pooling for classification
        pooled_causal = global_mean_pool(causal_output_nodes, batch)
        pooled_counter = global_mean_pool(counter_output_nodes, batch)

        # Causal and Counterfactual predictions
        causal_pre = self.fc(pooled_causal)
        counter_pre = self.fc(pooled_counter)

        # Original prediction y_pre (detached classification head)
        detached_weight = self.fc.weight.detach()
        detached_bias = self.fc.bias.detach() if self.fc.bias is not None else None
        y_pre = torch.nn.functional.linear(pooled_x, detached_weight, detached_bias)

        return causal_pre, counter_pre, y_pre, z_nodes, pooled_causal, pooled_x


class SynModel(torch.nn.Module):
    def __init__(self, num_embeddings, num_tasks, codebook_embedding_dim, codebook_encoder, representative_subgraphs,
                 layer_GNN,
                 JK="last", gnn_type='gin'):
        super().__init__()

        self.pool_layer = TopKPoolModel(num_layer=layer_GNN,
                                        emb_dim=codebook_embedding_dim,
                                        JK=JK, gnn_type=gnn_type, atom_encode=False, vir=False, drop_ratio=0.3)

        self.codebook_layer = VectorQuantizer(num_embeddings, codebook_embedding_dim, codebook_embedding_dim)
        self.cls = ClassLayer(codebook_embedding_dim, num_tasks)
        self.decoder = SimilarityDecoder(temperature=0.1)
        self.codebook_encoder = codebook_encoder
        self.representative_subgraphs = representative_subgraphs
        self.subgraph_batch = Batch.from_data_list(self.representative_subgraphs)
        self.node_encoder = torch.nn.Linear(4, codebook_embedding_dim)

    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != torch.long and self.pool_layer.gnn_node.atom_encoder.__class__.__name__ == 'AtomEncoder':
                pass

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dtype != torch.long and hasattr(self.pool_layer.gnn_node, 'convs'):

                first_conv = self.pool_layer.gnn_node.convs[0]
                if hasattr(first_conv, 'bond_encoder') and first_conv.bond_encoder.__class__.__name__ == 'BondEncoder':
                    pass
        data.x = self.node_encoder(data.x)
        data.y.long()
        data.edge_index.long()
        data.edge_attr.long()
        data_pool_x, data_pool_batch = self.pool_layer(data)

        device = data.x.device
        subgraph_batch_on_device = self.subgraph_batch.to(device)
        codebook = self.codebook_encoder(subgraph_batch_on_device)

        if data_pool_x is None or data_pool_x.numel() == 0:
            num_graphs_in_batch = data.num_graphs
            num_tasks = self.cls.fc.out_features

            # Need a dummy call to the codebook layer to get counter_matrix
            _, _, _, causal_matrix_dummy, counter_matrix_dummy = self.codebook_layer(codebook)

            causal_pre_dummy = torch.zeros(num_graphs_in_batch, num_tasks, device=device)
            counter_pre_dummy = torch.zeros_like(causal_pre_dummy)
            y_pre_dummy = torch.zeros_like(causal_pre_dummy)

            A_ori_list_dummy = []
            A_rec_list_dummy = []

            return (causal_pre_dummy, counter_pre_dummy, y_pre_dummy,
                    A_ori_list_dummy, A_rec_list_dummy,
                    torch.empty(0, self.cls.fc.in_features, device=device),  # causal_output (nodes)
                    torch.empty(0, self.cls.fc.in_features, device=device),  # data_pool_x
                    causal_matrix_dummy, counter_matrix_dummy,
                    None, None)  # pooled_causal, pooled_x

        A_ori_list = self.decoder(data_pool_x, data_pool_batch)

        processed_codebook, causal_cb, counter_cb, causal_matrix, counter_matrix = self.codebook_layer(
            codebook.to(device))
        causal_pre, counter_pre, y_pre, causal_output_nodes, pool_causal, pool_x = self.cls(data_pool_x,
                                                                                            data_pool_batch,
                                                                                            processed_codebook,
                                                                                            causal_cb, counter_cb)

        A_rec_list = self.decoder(causal_output_nodes, data_pool_batch)

        return causal_pre, counter_pre, y_pre, A_ori_list, A_rec_list, causal_output_nodes, data_pool_x, causal_matrix, counter_matrix, pool_causal, pool_x


class OgbModel(torch.nn.Module):
    def __init__(self, num_embeddings, num_tasks, codebook_encoder, codebook_embedding_dim, representative_subgraphs,
                 layer_GNN, JK="last",
                 gnn_type='gin'):
        super().__init__()

        self.pool_layer = TopKPoolModel(num_layer=layer_GNN,
                                        emb_dim=codebook_embedding_dim,
                                        JK=JK, gnn_type=gnn_type, atom_encode=True, vir=False)

        self.codebook_layer = VectorQuantizer(num_embeddings, codebook_embedding_dim, codebook_embedding_dim)
        self.cls = ClassLayer(codebook_embedding_dim, num_tasks)
        self.decoder = SimilarityDecoder(temperature=0.1)
        self.codebook_encoder = codebook_encoder
        self.representative_subgraphs = representative_subgraphs
        self.subgraph_batch = Batch.from_data_list(self.representative_subgraphs)

    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != torch.long and self.pool_layer.gnn_node.atom_encoder.__class__.__name__ == 'AtomEncoder':
                pass

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dtype != torch.long and hasattr(self.pool_layer.gnn_node, 'convs'):

                first_conv = self.pool_layer.gnn_node.convs[0]
                if hasattr(first_conv, 'bond_encoder') and first_conv.bond_encoder.__class__.__name__ == 'BondEncoder':
                    pass
        data.x.long()
        data.y.long()
        data.edge_index.long()
        data.edge_attr.long()
        data_pool_x, data_pool_batch = self.pool_layer(data)
        device = data.x.device
        subgraph_batch_on_device = self.subgraph_batch.to(device)
        codebook = self.codebook_encoder(subgraph_batch_on_device)
        if data_pool_x is None or data_pool_x.numel() == 0:
            num_graphs_in_batch = data.num_graphs
            num_tasks = self.cls.fc.out_features

            # Need a dummy call to the codebook layer to get causal/counter_matrix
            _, _, _, causal_matrix_dummy, counter_matrix_dummy = self.codebook_layer(codebook)

            causal_pre_dummy = torch.zeros(num_graphs_in_batch, num_tasks, device=device)
            counter_pre_dummy = torch.zeros_like(causal_pre_dummy)
            y_pre_dummy = torch.zeros_like(causal_pre_dummy)

            A_ori_list_dummy = []
            A_rec_list_dummy = []

            return (causal_pre_dummy, counter_pre_dummy, y_pre_dummy,
                    A_ori_list_dummy, A_rec_list_dummy,
                    torch.empty(0, self.cls.fc.in_features, device=device),  # causal_output (nodes)
                    torch.empty(0, self.cls.fc.in_features, device=device),  # data_pool_x
                    causal_matrix_dummy, counter_matrix_dummy,
                    None, None)  # pooled_causal, pooled_x

        A_ori_list = self.decoder(data_pool_x, data_pool_batch)
        processed_codebook, causal_cb, counter_cb, causal_matrix, counter_matrix = self.codebook_layer(
            codebook.to(device))

        causal_pre, counter_pre, y_pre, causal_output_nodes, pooled_causal, pooled_x = self.cls(data_pool_x,
                                                                                                data_pool_batch,
                                                                                                processed_codebook,
                                                                                                causal_cb, counter_cb)

        A_rec_list = self.decoder(causal_output_nodes, data_pool_batch)

        return causal_pre, counter_pre, y_pre, A_ori_list, A_rec_list, causal_output_nodes, data_pool_x, causal_matrix, counter_matrix, pooled_causal, pooled_x


class MniModel(torch.nn.Module):
    def __init__(self, num_embeddings, num_tasks, codebook_embedding_dim, codebook_encoder, representative_subgraphs,
                 layer_GNN,
                 JK="last", gnn_type='gin'):
        super().__init__()

        self.pool_layer = TopKPoolModel(num_layer=layer_GNN,
                                        emb_dim=codebook_embedding_dim,
                                        JK=JK, gnn_type=gnn_type, atom_encode=False, vir=False)

        self.codebook_layer = VectorQuantizer(num_embeddings, codebook_embedding_dim, codebook_embedding_dim)
        self.cls = ClassLayer(codebook_embedding_dim, num_tasks)
        self.decoder = SimilarityDecoder(temperature=0.1)
        self.codebook_encoder = codebook_encoder
        self.representative_subgraphs = representative_subgraphs
        self.subgraph_batch = Batch.from_data_list(self.representative_subgraphs)
        self.node_encoder = torch.nn.Linear(5, codebook_embedding_dim)

    def forward(self, data):
        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != torch.long and self.pool_layer.gnn_node.atom_encoder.__class__.__name__ == 'AtomEncoder':
                pass

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if hasattr(self.pool_layer.gnn_node, 'convs'):
                first_conv = self.pool_layer.gnn_node.convs[0]
                if hasattr(first_conv, 'bond_encoder') and first_conv.bond_encoder.__class__.__name__ == 'BondEncoder':
                    if data.edge_attr.dtype != torch.long:
                        pass
        data.x = self.node_encoder(data.x)
        data.y.long()
        data.edge_index.long()
        # The following two lines might need adjustment based on dataset specifics
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.long()
            if data.edge_attr.ndim == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)

        data_pool_x, data_pool_batch = self.pool_layer(data)

        device = data.x.device
        subgraph_batch_on_device = self.subgraph_batch.to(device)
        codebook = self.codebook_encoder(subgraph_batch_on_device)

        if data_pool_x is None or data_pool_x.numel() == 0:
            num_graphs_in_batch = data.num_graphs
            num_tasks = self.cls.fc.out_features

            # Need a dummy call to the codebook layer to get causal/counter_matrix
            _, _, _, causal_matrix_dummy, counter_matrix_dummy = self.codebook_layer(codebook)

            causal_pre_dummy = torch.zeros(num_graphs_in_batch, num_tasks, device=device)
            counter_pre_dummy = torch.zeros_like(causal_pre_dummy)
            y_pre_dummy = torch.zeros_like(causal_pre_dummy)

            A_ori_list_dummy = []
            A_rec_list_dummy = []

            return (causal_pre_dummy, counter_pre_dummy, y_pre_dummy,
                    A_ori_list_dummy, A_rec_list_dummy,
                    torch.empty(0, self.cls.fc.in_features, device=device),  # causal_output (nodes)
                    torch.empty(0, self.cls.fc.in_features, device=device),  # data_pool_x
                    causal_matrix_dummy, counter_matrix_dummy,
                    None, None)

        A_ori_list = self.decoder(data_pool_x, data_pool_batch)

        processed_codebook, causal_cb, counter_cb, causal_matrix, counter_matrix = self.codebook_layer(codebook)

        causal_pre, counter_pre, y_pre, causal_output_nodes, pooled_causal, pooled_x = self.cls(data_pool_x,
                                                                                                data_pool_batch,
                                                                                                processed_codebook,
                                                                                                causal_cb, counter_cb)

        A_rec_list = self.decoder(causal_output_nodes, data_pool_batch)

        return causal_pre, counter_pre, y_pre, A_ori_list, A_rec_list, causal_output_nodes, data_pool_x, causal_matrix, counter_matrix, pooled_causal, pooled_x


class MoModel(torch.nn.Module):
    def __init__(self, num_embeddings, num_tasks, codebook_embedding_dim, codebook_encoder, representative_subgraphs,
                 layer_GNN, node_feature_dim, edge_feature_dim, JK="last", gnn_type='gin'):
        super().__init__()

        self.pool_layer = MoTopKPoolModel(num_layer=layer_GNN,
                                          emb_dim=codebook_embedding_dim,
                                          edge_dim=edge_feature_dim,
                                          JK=JK, gnn_type=gnn_type, atom_encode=False)

        self.codebook_layer = VectorQuantizer(num_embeddings, codebook_embedding_dim, codebook_embedding_dim)
        self.cls = ClassLayer(codebook_embedding_dim, num_tasks)
        self.decoder = SimilarityDecoder(temperature=0.1)

        self.codebook_encoder = codebook_encoder
        self.representative_subgraphs = representative_subgraphs
        self.subgraph_batch = Batch.from_data_list(self.representative_subgraphs)
        self.node_encoder = torch.nn.Linear(node_feature_dim, codebook_embedding_dim)

    def forward(self, data):
        data.x = self.node_encoder(data.x.float())
        data.y = data.y.long()
        data.edge_index = data.edge_index.long()

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()

        data_pool_x, data_pool_batch = self.pool_layer(data)

        device = data.x.device
        subgraph_batch_on_device = self.subgraph_batch.to(device)
        codebook = self.codebook_encoder(subgraph_batch_on_device)

        if data_pool_x is None or data_pool_x.numel() == 0:
            num_graphs_in_batch = data.num_graphs
            num_tasks = self.cls.fc.out_features

            # Need a dummy call to the codebook layer to get causal/counter_matrix
            _, _, _, causal_matrix_dummy, counter_matrix_dummy = self.codebook_layer(codebook)

            causal_pre_dummy = torch.zeros(num_graphs_in_batch, num_tasks, device=device)
            counter_pre_dummy = torch.zeros_like(causal_pre_dummy)
            y_pre_dummy = torch.zeros_like(causal_pre_dummy)
            A_ori_list_dummy, A_rec_list_dummy = [], []

            return (causal_pre_dummy, counter_pre_dummy, y_pre_dummy,
                    A_ori_list_dummy, A_rec_list_dummy,
                    torch.empty(0, self.cls.fc.in_features, device=device),
                    torch.empty(0, self.cls.fc.in_features, device=device),
                    causal_matrix_dummy, counter_matrix_dummy,
                    None, None)

        A_ori_list = self.decoder(data_pool_x, data_pool_batch)

        processed_codebook, causal_cb, counter_cb, causal_matrix, counter_matrix = self.codebook_layer(codebook)

        causal_pre, counter_pre, y_pre, causal_output_nodes, pooled_causal, pooled_x = self.cls(data_pool_x,
                                                                                                data_pool_batch,
                                                                                                processed_codebook,
                                                                                                causal_cb, counter_cb)

        A_rec_list = self.decoder(causal_output_nodes, data_pool_batch)

        return causal_pre, counter_pre, y_pre, A_ori_list, A_rec_list, causal_output_nodes, data_pool_x, causal_matrix, counter_matrix, pooled_causal, pooled_x