import os
import numpy as np
import torch as th
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering

from .. import config
from .. import helpers
IGNORE_IN_TOTAL = ("")

# 检测是否为图数据集
def _is_graph_dataset(dataset):
    """检测数据集是否为图数据集（torch_geometric Dataset或包装器）"""
    from torch_geometric.data import Dataset as PyGDataset
    # 检查是否是PyG Dataset
    if isinstance(dataset, PyGDataset):
        return True
    # 检查是否是GraphDatasetWithDynamicFeatures（包装器）
    if hasattr(dataset, 'dataset') and isinstance(getattr(dataset, 'dataset'), PyGDataset):
        return True
    return False

device = th.device("cuda" if th.cuda.is_available() else "cpu")
def calc_metrics(labels, pred, hidden_features=None):
    """
    Compute metrics.

    :param labels: Label tensor
    :type labels: th.Tensor
    :param pred: Predictions tensor
    :type pred: th.Tensor
    :param hidden_features: Hidden layer features for clustering metrics (optional)
    :type hidden_features: np.ndarray
    :return: Dictionary containing calculated metrics
    :rtype: dict
    """
    acc, cmat = helpers.ordered_cmat(labels, pred)
    
    # 计算 ARI（不需要标签对齐）
    ari = adjusted_rand_score(labels, pred)
    
    
    metrics = {
        "acc": acc,
        "cmat": cmat,
        "nmi": normalized_mutual_info_score(labels, pred, average_method="geometric"),
        "ari": ari,
    }
    
    # 计算聚类算法的指标（KMeans 和 Spectral Clustering）
    if hidden_features is not None and len(hidden_features) > 0:
        n_clusters = len(np.unique(labels))
        try:
            # 大数据集采样优化
            if len(hidden_features) > 10000:
                sample_size = 10000
                sample_indices = np.random.choice(len(hidden_features), size=sample_size, replace=False)
                features_for_clustering = hidden_features[sample_indices]
                labels_for_clustering = labels[sample_indices]
            else:
                features_for_clustering = hidden_features
                labels_for_clustering = labels
            
            # K-Means
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                km_pred = kmeans.fit_predict(features_for_clustering)
                aligned_km_pred = helpers.align_labels(labels_for_clustering, km_pred)
                metrics["kmeans_acc"] = np.mean(aligned_km_pred == labels_for_clustering)
                metrics["kmeans_nmi"] = normalized_mutual_info_score(labels_for_clustering, aligned_km_pred)
                metrics["kmeans_ari"] = adjusted_rand_score(labels_for_clustering, aligned_km_pred)
            except Exception as e:
                print(f"[ERROR] K-Means clustering failed: {e}", flush=True)
                metrics["kmeans_acc"] = 0.0
                metrics["kmeans_nmi"] = 0.0
                metrics["kmeans_ari"] = 0.0
            
            # Spectral Clustering
            try:
                import warnings

                n_samples = len(features_for_clustering)
                if n_samples <= 2:
                    raise ValueError("Too few samples for spectral clustering")

                # Try larger neighborhood sizes to avoid disconnected kNN graphs.
                candidate_neighbors = [10, 20, 30, 50, int(np.sqrt(n_samples))]
                candidate_neighbors = sorted({k for k in candidate_neighbors if 2 <= k < n_samples})
                if not candidate_neighbors:
                    candidate_neighbors = [min(10, n_samples - 1)]

                # Using a larger neighborhood generally reduces disconnected components.
                n_neighbors = max(candidate_neighbors)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Graph is not fully connected, spectral embedding may not work as expected\.?",
                    )
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        random_state=42,
                        affinity='nearest_neighbors',
                        n_neighbors=n_neighbors,
                        assign_labels='kmeans',
                        n_init=10,
                    )
                    sc_pred = spectral.fit_predict(features_for_clustering)
                aligned_sc_pred = helpers.align_labels(labels_for_clustering, sc_pred)
                metrics["spectral_acc"] = np.mean(aligned_sc_pred == labels_for_clustering)
                metrics["spectral_nmi"] = normalized_mutual_info_score(labels_for_clustering, aligned_sc_pred)
                metrics["spectral_ari"] = adjusted_rand_score(labels_for_clustering, aligned_sc_pred)
            except Exception as e:
                print(f"[ERROR] Spectral Clustering failed: {e}", flush=True)
                metrics["spectral_acc"] = 0.0
                metrics["spectral_nmi"] = 0.0
                metrics["spectral_ari"] = 0.0
        except Exception as e:
            print(f"[ERROR] Clustering metrics calculation failed: {e}", flush=True)
    
    return metrics


def get_log_params(net):
    """Get the network parameters we want to log. The MUTAG-only CUE model has no fusion/attention/discriminator params, so this returns an empty dict."""
    return {}


def get_eval_data(dataset, n_eval_samples, batch_size, seed=None, random_mode=False):
    """
    Create a dataloader to use for evaluation

    :param dataset: Inout dataset.
    :type dataset: th.utils.data.Dataset
    :param n_eval_samples: Number of samples to include in the evaluation dataset. Set to None to use all available
                           samples.
    :type n_eval_samples: int
    :param batch_size: Batch size used for training.
    :type batch_size: int
    :return: Evaluation dataset loader
    :rtype: th.utils.data.DataLoader
    """
    # 检查是否为图数据集
    is_graph = _is_graph_dataset(dataset)
    
    if is_graph:
        # 图数据集：使用PyG DataLoader
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from src.data.load import collate_fn_with_laplacian
        generator = helpers.make_torch_generator(seed if seed is not None else 0, random_mode=random_mode, offset=10_000)
        eval_loader = PyGDataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
                                   drop_last=False, pin_memory=False, collate_fn=collate_fn_with_laplacian,
                                   generator=generator)
    return eval_loader


def batch_predict(net, eval_data, cfg, batch_size, calc_losses=True):
    """
    Compute predictions for `eval_data` in batches. Batching does not influence predictions, but it influences the loss
    computations.

    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloader
    :type eval_data: th.utils.data.DataLoader
    :param cfg: Experiment config
    :param batch_size: Batch size
    :type batch_size: int
    :param calc_losses: Whether to calculate losses during prediction
    :return: Label tensor, predictions tensor, list of dicts with loss values, array containing mean and std of cluster
             sizes, evidence_sums_total, hidden_features.
    :rtype:
    """
    labels = []
    predictions = []
    losses = []
    evidence_sums = []  # 每个簇的证据总和
    hidden_features = []  # 隐藏层特征用于聚类指标计算

    net.eval()
    # 检测是否为图数据加载器
    is_graph_loader = 'torch_geometric' in str(type(eval_data))
    
    with th.no_grad():
        for i, data in enumerate(eval_data):
            if is_graph_loader:
                # 图数据：data是PyG Batch对象
                batch_data = data.to(device)

                label = batch_data.y

                # 为多个视图提供同样的图批次，模型内部会处理（例如在eval模式下padding到N个视图）
                num_views = cfg.network_config.fusion_config.n_views
                batch = [batch_data] * num_views

            pred = net(batch)  # (B, K) - 证据
            pred_labels = pred.argmax(axis=1)  # (B,) - 预测类别

            # EDL概率计算: p = (e + 1) / S
            alpha = pred + 1
            S = alpha.sum(dim=1, keepdim=True)
            prob = alpha / S

            labels.append(helpers.npy(label))
            predictions.append(helpers.npy(pred_labels))
            
            # 收集隐藏特征（如果可用）
            if hasattr(net, 'hidden') and net.hidden is not None:
                hidden_features.append(helpers.npy(net.hidden))

            if calc_losses and label.size(0) == batch_size:
                batch_losses = net.calc_losses(ignore_in_total=IGNORE_IN_TOTAL)
                losses.append(helpers.npy(batch_losses))
                
            # 无论是否计算评估损失，都记录证据总和统计信息
            evidence_sums.append(helpers.npy(pred.sum(dim=0)))

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    hidden_features_array = np.concatenate(hidden_features, axis=0) if hidden_features else None
    net.train()
    
    # 从网络获取真实的类别数
    if hasattr(net, 'cfg') and hasattr(net.cfg, 'cm_config'):
        n_clusters = net.cfg.cm_config.n_clusters
    elif hasattr(net, 'evidence_fused') and net.evidence_fused is not None:
        n_clusters = net.evidence_fused.size(1)
    else:
        n_clusters = max(predictions.max() + 1, len(np.unique(predictions)))
    
    # ⚡ 优化：使用 bincount 代替循环
    cluster_sizes_total = np.bincount(predictions, minlength=n_clusters)
    
    # 聚合evidence统计量
    evidence_sums_total = np.array(evidence_sums).sum(axis=0) if evidence_sums else np.zeros(n_clusters)
    
    return labels, predictions, losses, cluster_sizes_total, evidence_sums_total, hidden_features_array


def get_logs(cfg, net, eval_data, iter_losses=None, epoch=None, include_params=True):
    if iter_losses is not None:
        logs = helpers.add_prefix(helpers.dict_means(iter_losses), "iter_losses")
    else:
        logs = {}

    if (epoch is None) or ((epoch % cfg.eval_interval) == 0):
        # ⚡ 性能优化: 如果已有训练loss(iter_losses)，评估时跳过昂贵的loss计算
        should_calc_eval_loss = (iter_losses is None)
        labels, pred, eval_losses, cluster_sizes, evidence_sums, hidden_features = batch_predict(
            net, eval_data, cfg, cfg.batch_size, calc_losses=should_calc_eval_loss
        )
        
        if should_calc_eval_loss:
            eval_losses = helpers.dict_means(eval_losses)
            logs.update(helpers.add_prefix(eval_losses, "eval_losses"))

        logs.update(helpers.add_prefix(calc_metrics(labels, pred, hidden_features), "metrics"))
        logs.update(helpers.add_prefix({"mean": cluster_sizes.mean(), "sd": cluster_sizes.std()}, "cluster_size"))
        logs.update(helpers.add_prefix({"mean": evidence_sums.mean(), "sd": evidence_sums.std()}, "evidence"))

    if include_params:
        logs.update(helpers.add_prefix(get_log_params(net), "params"))
    if epoch is not None:
        logs["epoch"] = epoch

    return logs


def eval_run(cfg, cfg_name, experiment_identifier, run, net, eval_data, callbacks=tuple(), load_best=True):
    """
    Evaluate a training run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param cfg_name: Config name
    :type cfg_name: str
    :param experiment_identifier: 8-character unique identifier for the current experiment
    :type experiment_identifier: str
    :param run: Run to evaluate
    :type run: int
    :param net: Model
    :type net:
    :param eval_data: Evaluation dataloder
    :type eval_data: th.utils.data.DataLoader
    :param callbacks: List of callbacks to call after evaluation
    :type callbacks: List
    :param load_best: Load the "best.pt" model before evaluation?
    :type load_best: bool
    :return: Evaluation logs
    :rtype: dict
    """
    if load_best:
        model_path = helpers.get_save_dir(
            cfg_name,
            experiment_identifier,
            run,
            dataset_name=cfg.dataset_config.name,
        ) / "best.pt"
        if os.path.isfile(model_path):
            net.load_state_dict(th.load(model_path))
        else:
            print(f"Unable to load best model for evaluation. Model file not found: {model_path}")
    logs = get_logs(cfg, net, eval_data, include_params=True)
    for cb in callbacks:
        cb.at_eval(net=net, logs=logs)
    return logs
