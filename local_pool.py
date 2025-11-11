import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import DenseGCNConv, dense_diff_pool, DenseGATConv, DenseGINConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

import torch.nn as nn
import torch.nn.functional as F


class StructPoolLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # GNN 生成节点嵌入
        self.embed_gnn = DenseGCNConv(in_channels, hidden_channels)
        self.embed_gnn_1 = DenseGCNConv(hidden_channels, hidden_channels)

        # GNN 生成聚类分配矩阵
        self.pool_gnn = DenseGCNConv(hidden_channels, num_clusters)

        # 定义 Dropout 层
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, x, adj, mask=None):
        # 1. 计算节点嵌入（第一层）
        x_embed = F.relu(self.embed_gnn(x, adj, mask))  # [batch, max_nodes, hidden]
        x_embed = self.dropout_layer(x_embed)  # 添加 Dropout

        # 2. 计算节点嵌入（第二层）
        x_embed = F.relu(self.embed_gnn_1(x_embed, adj, mask))
        x_embed = self.dropout_layer(x_embed)  # 添加 Dropout

        # 3. 生成分配矩阵（Softmax 前 Dropout）
        s = self.pool_gnn(x_embed, adj, mask)
        s = self.dropout_layer(s)  # 添加 Dropout
        s = F.softmax(s, dim=-1)  # [batch, max_nodes, clusters]

        # 4. 可微分池化
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(
            x_embed, adj, s, mask
        )

        return x_pool, adj_pool, link_loss, ent_loss


if __name__ == "__main__":
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()

    # 图2: 4个节点，特征维度2
    x2 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]], dtype=torch.float)
    edge_index2 = torch.tensor([[0, 2], [1, 3]], dtype=torch.long).t().contiguous()

    # 创建Data对象并合并为批量
    batch = Batch.from_data_list([
        Data(x=x1, edge_index=edge_index1),
        Data(x=x2, edge_index=edge_index2)
    ])
    # 将稀疏邻接矩阵转换为密集格式 [batch_size, max_nodes, max_nodes]
    adj = to_dense_adj(batch.edge_index, batch=batch.batch)

    # 将节点特征对齐为密集格式 [batch_size, max_nodes, in_channels]
    x_dense, _ = to_dense_batch(batch.x, batch=batch.batch)

    # 提取关键参数
    batch_size = adj.size(0)
    in_channels = batch.x.size(1)
    # 参数设置
    hidden_channels = 16
    num_clusters = 4

    # 初始化模型
    model = StructPoolLayer(in_channels, hidden_channels, num_clusters)
    # 前向传播
    x_pool, adj_pool, _, _ = model(x_dense, adj)
    print("Pooled Node Features (Batch):", x_pool.shape)  # [batch_size, num_clusters, hidden_channels]
    print("Pooled Adjacency Matrix (Batch):", adj_pool.shape)  # [batch_size, num_clusters, num_clusters]

    batch_size, num_clusters = x_pool.shape[0], x_pool.shape[1]
    data_list = []

    for i in range(batch_size):
        x_i = x_pool[i]  # [num_clusters, hidden], 保持梯度
        adj_i = adj_pool[i]  # [num_clusters, num_clusters], 需要梯度的源

        # 构造全连接的边索引 (模拟邻接矩阵结构)
        rows, cols = torch.arange(num_clusters).repeat_interleave(num_clusters), torch.arange(num_clusters).repeat(
            num_clusters)
        edge_index = torch.stack([rows, cols], dim=0)  # [2, num_clusters*num_clusters]

        # 将邻接矩阵值作为边权重传递梯度
        edge_attr = adj_i.view(-1, 1)  # [num_clusters^2, 1]

        data = Data(x=x_i, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # 验证梯度
    loss = (batch.x ** 2).mean()  # 示例损失函数
    loss.backward()
    # 输出结果
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"参数名: {name}, 梯度形状: {param.grad.shape}, 梯度范数: {param.grad.norm().item()}")
        else:
            print(f"参数名: {name} 的梯度未计算（可能未参与计算或未反向传播）")

    print("Batch 对象节点特征:", batch)  # [batch_size * num_clusters, hidden_channels]
    print("Batch 的边索引:", batch.edge_index.shape)  # [2, total_edges]
    print("Batch 的批次属性:", batch.batch.shape)  # [batch_size * num_clusters]
