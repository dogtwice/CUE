"""
图神经网络Backbone实现

支持的GNN类型:
- GIN (Graph Isomorphism Network): 侧重结构特征
- GCN (Graph Convolutional Network): 侧重属性特征

这些模型可以端到端训练，参数参与聚类损失的优化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool


class GraphBackbone(nn.Module):
    """图神经网络Backbone基类"""
    
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_size = [cfg.hidden_dim]  # 必须设置output_size供上层模块使用
    
    def forward(self, batch_data):
        """
        Args:
            batch_data: PyG Batch对象或Data对象
                - x: (total_nodes, node_feat_dim)
                - edge_index: (2, total_edges)
                - edge_attr: (total_edges, edge_feat_dim) [可选]
                - batch: (total_nodes,) 节点到图的映射
        
        Returns:
            graph_embeddings: (batch_size, hidden_dim)
        """
        raise NotImplementedError


class GIN(GraphBackbone):
    """
    Graph Isomorphism Network (GIN)
    
    特点:
    - 强大的图结构表达能力
    - Sum aggregation捕获邻域信息
    - 适合提取拓扑特征
    """
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        self.num_layers = cfg.num_layers
        self.hidden_dim = cfg.hidden_dim
        self.use_edge_attr = cfg.use_edge_attr
        self.dropout = cfg.dropout
        
        # GIN卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(self.num_layers):
            if i == 0:
                # 第一层：输入维度 -> 隐藏维度
                mlp = nn.Sequential(
                    nn.Linear(cfg.input_node_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # 图级池化
        if cfg.pooling == "mean":
            self.pool = global_mean_pool
        elif cfg.pooling == "sum":
            self.pool = global_add_pool
        elif cfg.pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling type: {cfg.pooling}")
    
    def forward(self, batch_data):
        """
        Args:
            batch_data: PyG Batch对象
        
        Returns:
            graph_embeddings: (batch_size, hidden_dim)
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch = batch_data.batch if hasattr(batch_data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GNN消息传递
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级池化
        graph_embed = self.pool(x, batch)  # (batch_size, hidden_dim)
        
        return graph_embed
