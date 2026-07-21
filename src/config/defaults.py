from typing import Tuple, List, Union, Optional
from typing_extensions import Literal

from .config import Config


class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str


class Loss(Config):
    # Number of clusters
    n_clusters: int = 2
    # Terms to use in the loss, separated by '|'. E.g. "GC3|UCL"
    # Available loss terms:
    #   - GC1: Geometric structural alignment (L_GC1)
    #   - GC2: Geometric partition separability (L_GC2)
    #   - GC3: Geometric distribution compactness (L_GC3)
    #   - GCL: Graph Contrastive Learning loss (simplified)
    #   - RC: Reliability-aware consensus loss (L_RC)
    #   - UCL: Uncertainty-calibrated contrastive loss (L_UCL)
    funcs: str
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Tuple[Union[float, int], ...] = None
    # Multiplication factor for the sigma hyperparameter
    rel_sigma: float = 0.15
    tau: float = 0.1
    tau_alpha: float = 0.3  # dynamic temperature coefficient (lambda)
    # Fraction of batch size to use as the number of negative samples in the contrastive loss.
    negative_samples_ratio: float = 1
    # Similarity function for the contrastive loss. "cos" (default) and "gauss" are supported.
    contrastive_similarity: Literal["cos", "gauss"] = "cos"

class Optimizer(Config):
    # Base learning rate
    learning_rate: float = 0.001
    # Max gradient norm for gradient clipping.
    clip_norm: float = 5.0
    # Step size for the learning rate scheduler. None disables the scheduler.
    scheduler_step_size: int = None
    # Multiplication factor for the learning rate scheduler
    scheduler_gamma: float = 0.1


class EvidenceHead(Config):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden: int = 64
    # Use batch norm after the first fully connected layer?
    use_bn: bool = True


class GIN(Config):
    """Graph Isomorphism Network配置"""
    # 节点特征维度
    input_node_dim: int = 7
    # 边特征维�?(可�?
    input_edge_dim: int = 4
    # 隐藏层维�?
    hidden_dim: int = 64
    # GNN层数
    num_layers: int = 3
    # 图池化方�? 'mean', 'sum', 'max'
    pooling: Literal["mean", "sum", "max"] = "sum"
    # 是否使用边特�?
    use_edge_attr: bool = False
    # Dropout�?
    dropout: float = 0.0


class Augmentation(Config):
    """Graph augmentation configuration."""
    node_drop_rate: float = 0.0
    edge_drop_rate: float = 0.1
    feat_mask_rate: float = 0.1
class Fusion(Config):
    # Number of views in the dataset
    n_views: int


class CUE(Config):
    # Tuple of encoder configs. One for each modality.
    backbone_configs: Tuple[GIN, ...]
    # Fusion module config.
    fusion_config: Fusion
    # Clustering module config.
    cm_config: Optional[EvidenceHead]
    # 图增强配�?
    augmentation_config: Optional[Augmentation] = None
    # Loss function config
    loss_config: Loss
    # Optimizer config
    optimizer_config: Optimizer = Optimizer()


class Experiment(Config):
    # Dataset config
    dataset_config: Dataset
    # Model config
    network_config: CUE
    # Number of training runs
    n_runs: int = 1
    # Number of training epochs
    n_epochs: int = 40
    # Batch size
    batch_size: int = 128
    # Number of epochs between model evaluation.
    eval_interval: int = 20
    # Number of epochs between model checkpoints.
    checkpoint_interval: int = 200
    # Patience for early stopping.
    patience: int = 50000
    # Number of samples to use for evaluation. Set to None to use all samples in the dataset.
    n_eval_samples: int = None
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term: str ='tot' #early stopping和保存模型的依据, 比如选择tot最小的模型保存, 或者GC1最小的保存, 可选GC1,GC2,GC3,RC,tot
    seed: int = 0
    # Enable random mode? When True, random seed will not be set.
    random: bool = False


