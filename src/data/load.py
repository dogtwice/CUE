"""
图数据集加载 - 使用torch_geometric官方TUDataset

只支持图数据集，使用torch_geometric.datasets.TUDataset加载官方数据集
支持动态特征处理（无节点特征时自动合成OneHotDegree）

MUTAG-only release: 仅 GIN 视图，无 ChebNet/Laplacian 预计算。
"""

from .. import config
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def load_dataset(name, **kwargs):
    """
    加载图数据集。

    Args:
        name: 数据集名称（如 'MUTAG'），支持 torch_geometric.datasets.TUDataset。
        **kwargs: 其他参数（暂未使用）。

    Returns:
        GraphDatasetWithDynamicFeatures 包装的 TUDataset 对象，支持动态特征处理
    """
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(
        root=config.DATA_DIR / "raw",
        name=name,
        use_node_attr=True
    )

    # 包装dataset以支持动态特征处理
    dataset = GraphDatasetWithDynamicFeatures(dataset)

    return dataset


class GraphDatasetWithDynamicFeatures:
    """
    包装TUDataset，提供动态特征处理（方案D优化版）

    对于无节点特征的数据集（如IMDB-BINARY），在__init__时一次性计算
    OneHotDegree特征，而不是每次forward都算（比之前的动态方案快很多）
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._has_node_features = self._check_has_features()
        self._max_degree = None  # 缓存全局最大度数
        self._feature_cache = {}  # 缓存预计算的 OneHotDegree 特征

        # 如果没有节点特征，预计算并缓存所有 OneHotDegree 特征
        if not self._has_node_features:
            self._precompute_and_cache_features()

    def _check_has_features(self):
        """检查数据集是否有节点特征"""
        sample = self.dataset[0]
        has_features = hasattr(sample, 'x') and sample.x is not None and sample.x.numel() > 0
        if has_features:
            print(f"[Dataset] Detected node features: shape={sample.x.shape}")
        else:
            print(f"[Dataset] No node features detected, will generate OneHotDegree")
        return has_features

    def _precompute_and_cache_features(self):
        """预计算并缓存所有 OneHotDegree 特征（一次性，快速）"""
        # 第一步：扫描整个数据集，找到全局最大度数
        max_degree = 0
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(deg.max()))

        self._max_degree = max_degree
        num_classes = max_degree + 1
        print(f"[Dataset] Precomputing OneHotDegree features with {num_classes} classes (max_degree={max_degree})...")

        # 第二步：预计算并缓存所有特征（避免 __getitem__ 中的重复计算）
        from tqdm import tqdm
        for idx in tqdm(range(len(self.dataset)), desc="Caching features"):
            data = self.dataset[idx]
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            # 预计算 OneHotDegree 并缓存到内存
            self._feature_cache[idx] = F.one_hot(deg, num_classes=num_classes).float()

        print(f"[Dataset] Cached {len(self._feature_cache)} OneHotDegree features")

    def __getitem__(self, idx):
        """直接返回数据，从缓存读取预计算的特征（极快）"""
        data = self.dataset[idx]

        # 如果原始数据没有节点特征，从缓存读取预计算的 OneHotDegree
        if not self._has_node_features and idx in self._feature_cache:
            data.x = self._feature_cache[idx]

        return data

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    @property
    def num_node_features(self):
        """获取节点特征维度"""
        if self._has_node_features:
            return self.dataset.num_node_features
        return self._max_degree + 1

    @property
    def num_edge_features(self):
        """获取边特征维度"""
        return self.dataset.num_edge_features

    @property
    def num_classes(self):
        """获取类别数"""
        return self.dataset.num_classes

    def __getattr__(self, name):
        """将属性请求转发给底层的TUDataset，并防止递归"""
        if name == 'dataset':
            raise AttributeError("'GraphDatasetWithDynamicFeatures' object has no attribute 'dataset'")
        return getattr(self.dataset, name)


def collate_fn_with_laplacian(batch):
    """自定义collate函数（保留接口名供 train/evaluate 调用）。MUTAG-only release 无 laplacian 预计算。"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
