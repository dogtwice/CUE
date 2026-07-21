"""
证据理论相关模块
包含证据生成、融合和不确定性建模
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import itertools


class EvidenceHead(nn.Module):
    """
    证据生成网络 - RBF 核版本
    
    使用 RBF (Radial Basis Function) 核计算证据：
    e_ik = γ * exp(-||h_i - μ_k||^2 / (2 * σ^2))
    
    其中：
    - h_i: 样本特征向量（来自隐藏层）
    - μ_k: 聚类中心（证据头权重，由 K-Means 初始化）
    - γ (gamma): 缩放因子，控制最高证据强度
    - σ (sigma): 核宽度，控制证据对距离的敏感度
    
    优势：
    1. 局部响应性：只有样本靠近中心时才产生强证据，更符合不确定度建模
    2. 严格非负：exp 函数天然输出正值证据
    """
    def __init__(self, input_dim, cfg):
        super().__init__()

        # 1. hidden特征提取层
        hidden_layers: List[nn.Module] = [nn.Linear(input_dim[0], cfg.n_hidden)]
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        
        # 2. 证据头：从hidden到证据空间的映射
        # 注意：此处 linear 仅利用其 weight 存储聚类中心 μ
        self.linear = nn.Linear(cfg.n_hidden, cfg.n_clusters, bias=False) 
        
        # 3. RBF 缩放因子 γ (gamma) - 控制证据的最大值
        # 如果 γ=10.0，在中心点样本的 S = 10 + K，对应 u = K/(10+K)
        self.register_buffer('gamma', torch.tensor(10))
        
        # 4. RBF 核宽度 σ (sigma) - 调节高斯核的“宽度”
        self.register_buffer('sigma', torch.tensor(0.75))
        
        self.n_clusters = cfg.n_clusters
        self.n_hidden = cfg.n_hidden
        
        
        # 执行证据头的核心初始化逻辑（必须在最后，以避免全局初始化覆盖）
        self._reset_evidence_head_parameters()

    def _reset_evidence_head_parameters(self):
        """
        初始化证据头参数的核心逻辑
        
        策略: 权重正交化 (Prototype Orthogonality)
        - 目的: 让代表不同簇的权重向量互斥，利于L_sep正交损失快速起效
        - 数学含义: <W_i, W_j> ≈ 0 for i != j
        - 注：在余弦相似度中，正交的权重意味着完全不同的聚类原型
        
        注意：gamma 和 bias 都是固定的超参数，不需要初始化
        """
        # 正交初始化权重
        nn.init.orthogonal_(self.linear.weight)

    def perform_spectral_init(self, features, random_state: Optional[int] = 42):
        """
        使用谱聚类初始化聚类中心 (evidence head weights)
        
        谱聚类基于特征的局部结构（通过最近邻亲和矩阵），相比K-Means更能捕捉复杂流形结构。
        这特别适合于有非凸聚类或流形几何特征的数据集。
        
        风险防范：
        1. 在 train 模式下提取特征，确保 BN 分布与训练阶段对齐
        2. 对大数据集进行采样，避免 O(n^2) 的谱聚类计算爆炸
        
        Args:
            features: (N, D) tensor, 应为hidden特征
            random_state: 随机种子
        """
        import numpy as np
        import warnings
        from sklearn.cluster import SpectralClustering, KMeans
        
        # Convert to numpy
        X = features.detach().cpu().numpy()
        n_samples = X.shape[0]
        
        # [风险防范] 对大数据集采样
        # 谱聚类的复杂度为 O(n^2 + n^3)（计算相似度矩阵 + 谱分解）
        # 当 n > 10000 时，直接运行会导致内存溢出或极端耗时
        # 策略：采样到 max_samples，在采样数据上运行谱聚类，然后用 KMeans 给剩余数据分配标签
        max_samples = 10000
        
        rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

        if n_samples > max_samples:
            # Step 1: 随机采样 max_samples 个样本
            sample_indices = rng.choice(n_samples, size=max_samples, replace=False)
            X_sampled = X[sample_indices]
            
            # Step 2: 在采样数据上运行谱聚类
            n_sampled = X_sampled.shape[0]
            candidate_neighbors = [10, 20, 30, 50, int(np.sqrt(n_sampled))]
            candidate_neighbors = sorted({k for k in candidate_neighbors if 2 <= k < n_sampled})
            if not candidate_neighbors:
                candidate_neighbors = [min(10, n_sampled - 1)]
            n_neighbors = max(candidate_neighbors)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Graph is not fully connected, spectral embedding may not work as expected\.?",
                )
                spectral = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                    assign_labels='kmeans',
                    n_init=10,
                    n_jobs=-1,
                )
                cluster_labels_sampled = spectral.fit_predict(X_sampled)
            
            # Step 3: 用 KMeans 在完整数据上快速拟合，使用谱聚类的结果作为初始化
            # 这样可以充分利用谱聚类找到的流形结构，同时加速最终的中心计算
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=5,
                random_state=random_state,
                max_iter=100
            )
            cluster_labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_
            
        else:
            # Step 1: 在完整数据上运行谱聚类
            candidate_neighbors = [10, 20, 30, 50, int(np.sqrt(n_samples))]
            candidate_neighbors = sorted({k for k in candidate_neighbors if 2 <= k < n_samples})
            if not candidate_neighbors:
                candidate_neighbors = [min(10, n_samples - 1)]
            n_neighbors = max(candidate_neighbors)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Graph is not fully connected, spectral embedding may not work as expected\.?",
                )
                spectral = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                    assign_labels='kmeans',
                    n_init=10,
                    n_jobs=-1,
                )
                cluster_labels = spectral.fit_predict(X)
            
            # Step 2: 计算每个聚类的中心（平均向量）
            centers = []
            for k in range(self.n_clusters):
                mask = cluster_labels == k
                if mask.any():
                    center_k = X[mask].mean(axis=0)
                else:
                    # 如果某个聚类为空，取全局均值
                    center_k = X.mean(axis=0)
                centers.append(center_k)
            centers = np.array(centers)
        
        # Update weights
        new_weights = torch.tensor(centers, dtype=self.linear.weight.dtype, device=self.linear.weight.device)
        self.linear.weight.data.copy_(new_weights)

    def forward(self, x):
        """
        前向传播 - RBF 证据生成
        
        步骤：
        1. 特征提取: x -> hidden
        2. 计算与各聚类中心的欧氏距离平方: ||h_i - μ_k||^2
        3. 计算 RBF 证据: γ * exp(-dist^2 / (2σ^2))
        4. 返回证据和隐藏层特征
        
        Args:
            x: 输入特征 (B, D)
            
        Returns:
            evidence: 非负证据 (B, K)
            hidden: 隐藏层特征 (B, n_hidden)
        """
        # 步骤1: 特征提取
        hidden = self.hidden(x)  # (B, n_hidden)
        
        # 步骤2: 计算欧氏距离平方 ||h_i - μ_k||^2
        # 利用 ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> 提高并行计算效率
        h_norm_sq = torch.sum(hidden**2, dim=1, keepdim=True)  # (B, 1)
        mu_norm_sq = torch.sum(self.linear.weight**2, dim=1, keepdim=True).t()  # (1, K)
        dist_sq = h_norm_sq + mu_norm_sq - 2 * torch.mm(hidden, self.linear.weight.t())
        
        # 精度修正，防止浮点误差导致负数距离
        dist_sq = F.relu(dist_sq)
        
        # 步骤3: 计算 RBF 证据
        # [原因分析] $u \approx 1$ 通常是因为高维空间距离 $\|h-μ\|^2$ 过大，且 $\gamma$ 过小
        # 1. 对距离进行维度归一化：dist_sq / n_hidden
        # 2. evidence = γ * exp(- (dist_sq/d) / (2σ^2))
        normalized_dist_sq = dist_sq / (self.n_hidden + 1e-10)
        sigma = self.sigma if isinstance(self.sigma, torch.Tensor) else torch.as_tensor(self.sigma)
        evi_scale = self.gamma if isinstance(self.gamma, torch.Tensor) else torch.as_tensor(self.gamma)
        exponent = -normalized_dist_sq / (2 * torch.square(sigma) + 1e-10)
        evidence = evi_scale * torch.exp(exponent)

        return evidence, hidden


class UDF(nn.Module):
    """
    Uncertainty-Driven Fusion (UDF) via generalized PCR6
    
    用于融合多个视图的证据
    
    Reference:
    @article{han2022trusted,
      title={Trusted multi-view classification with dynamic evidential fusion},
      author={Han, Zongbo and Zhang, Changqing and Fu, Huazhu and Zhou, Joey Tianyi},
      journal={IEEE transactions on pattern analysis and machine intelligence},
      volume={45},
      number={2},
      pages={2551--2566},
      year={2022},
      publisher={IEEE}
    }
    """
    def __init__(self, num_views, n_clusters):
        super(UDF, self).__init__()
        self.num_views = num_views
        self.n_clusters = n_clusters
        self.theta_index = n_clusters

    def _alpha_to_mass(self, alpha):
        """
        Convert Dirichlet parameters to singleton masses plus uncertainty mass.

        Returns:
            mass: (B, K + 1) where the last column is the mass on Theta.
        """
        S = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-9)
        evidence = alpha - 1
        beliefs = evidence / S
        uncertainty = self.n_clusters / S
        return torch.cat([beliefs, uncertainty], dim=1)

    def _pcr6_combine_multi(self, masses):
        """
        Generalized multi-source PCR6 over singleton focal elements plus Theta.

        This follows the appendix algorithm in `PCR6.pdf` directly on the
        focal-element combinations, which makes the result permutation
        invariant with respect to source order.
        """
        if not masses:
            raise ValueError("masses list is empty")
        if len(masses) == 1:
            return masses[0]

        batch_size = masses[0].shape[0]
        device = masses[0].device
        dtype = masses[0].dtype
        n_focals = self.n_clusters + 1
        fused = torch.zeros(batch_size, n_focals, device=device, dtype=dtype)

        for focal_indices in itertools.product(range(n_focals), repeat=len(masses)):
            local_conflict = torch.ones(batch_size, device=device, dtype=dtype)
            local_sum = torch.zeros(batch_size, device=device, dtype=dtype)
            selected_singletons = set()
            selected_masses = []

            for view_idx, focal_idx in enumerate(focal_indices):
                current_mass = masses[view_idx][:, focal_idx]
                selected_masses.append(current_mass)
                local_conflict = local_conflict * current_mass
                local_sum = local_sum + current_mass
                if focal_idx != self.theta_index:
                    selected_singletons.add(focal_idx)

            if len(selected_singletons) == 0:
                fused[:, self.theta_index] += local_conflict
                continue

            if len(selected_singletons) == 1:
                target_idx = next(iter(selected_singletons))
                fused[:, target_idx] += local_conflict
                continue

            # PCR6 conflict redistribution: each source contributes its share
            # back to the focal element that generated the conflict.
            denom = local_sum.clamp_min(1e-9)
            for focal_idx, source_mass in zip(focal_indices, selected_masses):
                fused[:, focal_idx] += source_mass * local_conflict / denom

        return fused

    def pcr6_combine(self, alpha):
        """
        融合多个视图的证据
        
        Args:
            alpha: 字典，{view_id: Dirichlet参数}
            
        Returns:
            alpha_a: 融合后的 Dirichlet 参数
            u: 融合后的不确定性
        """
        if len(alpha) == 0:
            raise ValueError("alpha dictionary is empty")
        
        ordered_keys = sorted(alpha.keys())
        masses = [self._alpha_to_mass(alpha[k]) for k in ordered_keys]
        fused_mass = self._pcr6_combine_multi(masses)

        b_a = fused_mass[:, :self.n_clusters]
        u_a = fused_mass[:, self.theta_index:self.theta_index + 1].clamp_min(1e-9)

        S_a = self.n_clusters / u_a
        e_a = b_a * S_a
        alpha_a = e_a + 1
        return alpha_a, u_a

    def forward(self, X, evidences):
        """
        前向传播
        
        Args:
            X: 输入数据列表（未使用，保留接口兼容性）
            evidences: 各视图的证据字典 {view_id: evidence}
            
        Returns:
            evidence_fused: 融合后的证据 (B, K)
            predicted: 预测类别 (B,)
            u: 不确定性 (B, 1)
        """
        # 将证据转换为 Dirichlet 参数
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidences[v_num] + 1
        
        # DS融合
        alpha_a, u = self.pcr6_combine(alpha)
        evidence_fused = alpha_a - 1
        
        # 预测
        _, predicted = torch.max(evidence_fused.data, 1)

        return evidence_fused, predicted, u
