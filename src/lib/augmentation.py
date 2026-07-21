import torch
from torch_geometric.data import Batch
from torch_geometric.utils import dropout_edge, dropout_node


def _apply_node_mask(data, node_mask):
    if node_mask is None:
        return data
    if node_mask.dtype != torch.bool:
        node_mask = node_mask.to(dtype=torch.bool)
    if hasattr(data, "x") and data.x is not None:
        if node_mask.numel() != data.x.size(0):
            raise ValueError(f"node_mask size {node_mask.numel()} does not match num_nodes {data.x.size(0)}")
        data.x = data.x[node_mask]
    data.num_nodes = int(node_mask.sum().item())
    return data


def graph_augment(
    data,
    node_drop_rate=0.0,
    edge_drop_rate=0.1,
    feat_mask_rate=0.1,
):
    """Clone and augment a PyG Data or Batch object."""
    if node_drop_rate <= 0 and edge_drop_rate <= 0 and feat_mask_rate <= 0:
        return data

    new_data = data.clone()

    if node_drop_rate > 0 and hasattr(new_data, "edge_index") and new_data.edge_index is not None:
        edge_index, _, node_mask = dropout_node(
            new_data.edge_index,
            p=float(node_drop_rate),
            num_nodes=new_data.num_nodes,
            training=True,
            relabel_nodes=True,
        )
        new_data.edge_index = edge_index
        new_data = _apply_node_mask(new_data, node_mask)

    if edge_drop_rate > 0 and hasattr(new_data, "edge_index") and new_data.edge_index is not None:
        edge_index, _ = dropout_edge(new_data.edge_index, p=float(edge_drop_rate), training=True)
        new_data.edge_index = edge_index

    if feat_mask_rate > 0 and hasattr(new_data, "x") and new_data.x is not None:
        mask = torch.bernoulli(torch.full_like(new_data.x, 1 - float(feat_mask_rate)))
        new_data.x = new_data.x * mask

    if hasattr(new_data, "x") and new_data.x is not None:
        new_data.num_nodes = int(new_data.x.size(0))

    return new_data
