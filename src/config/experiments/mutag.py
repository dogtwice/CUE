"""
MUTAG分子图数据集实验配置 - CUE (Conflict and Uncertainty-aware Evidential clustering)

数据集信息:
- 样本数: 188个分子图
- 类别数: 2 (致突变 vs 非致突变)

架构 (对齐论文):
- 视图: V=3 (1个原图 + 2个随机增强视图), 共享 GIN 编码器
- 融合: UDF (PCR6 比例冲突重分配) 生成样本级证据与不确定性
- 损失: L_GC1+L_GC2+L_GC3 (几何约束) + L_RC (可靠性共识) + L_UCL (不确定性校准对比)
- 证据头: RBF 核, gamma=10, sigma=0.75
"""
from ..defaults import (
    Experiment, EvidenceHead, Fusion, GIN,
    Loss, Dataset, CUE, Optimizer, Augmentation
)

mutag = Experiment(
    dataset_config=Dataset(
        name="MUTAG"
    ),
    network_config=CUE(
        backbone_configs=(
            # View 1: GIN - 共享 Backbone
            # 同时也作为增强视图的 Backbone
            GIN(
                input_node_dim=7,      # MUTAG节点特征维度
                input_edge_dim=4,      # MUTAG边特征维度
                hidden_dim=16,         # 隐藏层维度
                num_layers=5,          # GNN层数
                pooling="sum",         # Sum池化强调结构
                use_edge_attr=True,   # 使用边特征
                dropout=0.1            # 防止过拟合
            ),
        ),
        fusion_config=Fusion(n_views=3),
        cm_config=EvidenceHead(n_clusters=2, n_hidden=64),
        augmentation_config=Augmentation(
            node_drop_rate=0,
            edge_drop_rate=0.1,
            feat_mask_rate=0.1
        ),
        loss_config=Loss(
            funcs="GC1|GC2|GC3|RC|UCL",
            n_clusters=2,
            weights=(1, 1, 1, 0.1, 0.1),
            tau=0.1,
            tau_alpha=0.3,
            negative_samples_ratio=1
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-3,  # 学习率
        )
    ),
    n_epochs=100,
    n_runs=1,
    batch_size=64,
    seed=5,
    random=False,
)

