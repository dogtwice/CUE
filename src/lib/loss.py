import torch as th
import torch.nn as nn
from .. import config
from . import kernel

import torch
import torch.nn.functional as F

EPSILON = 1E-9
DEBUG_MODE = False
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def triu(X):

    return th.sum(th.triu(X, diagonal=1))

def _atleast_epsilon(X, eps=EPSILON):
    return th.where(X < eps, X.new_tensor(eps), X)

def d_cs(A, K, n_clusters):

    nom = th.t(A) @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)

    nom = _atleast_epsilon(nom)
    dnom_squared = _atleast_epsilon(dnom_squared, eps=EPSILON**2)

    d = 2 / (n_clusters * (n_clusters - 1)) * triu(nom / th.sqrt(dnom_squared))
    return d

class LossTerm:
    required_tensors = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, net, cfg, extra):
        raise NotImplementedError()

class GC1(LossTerm):
    required_tensors = ["hidden_kernel"]
    def __call__(self, net, cfg, extra):
        return (d_cs(net.evidence_fused, extra["hidden_kernel"], cfg.n_clusters))

class GC2(LossTerm):
    def __call__(self, net, cfg, extra):
        k = cfg.n_clusters
        A = net.evidence_fused  # Shape: (B, K)
        # Compute cluster correlation matrix: A^T @ A -> (K, K)
        cluster_corr = th.t(A) @ A
        # Enforce orthogonality on cluster dimension by minimizing upper triangle
        return (2 / (k * (k - 1)) * triu(cluster_corr))

class GC3(LossTerm):
    required_tensors = ["hidden_kernel"]

    def __init__(self, cfg):
        super().__init__()
        self.eye = th.eye(cfg.n_clusters, device=config.DEVICE)

    def __call__(self, net, cfg, extra):
        # 1. Get evidence (non-negative after ReLU)
        e = net.evidence_fused
        
        # 2. Convert to Dirichlet parameters
        alpha = e + 1.0
        
        # 3. [KEY] Normalize to probabilities/beliefs in [0, 1]
        # This ensures dimensional consistency with self.eye (also in [0, 1])
        prob = alpha / (th.sum(alpha, dim=1, keepdim=True) + 1e-8)
        
        # 4. Compute distance and similarity
        # Now prob=[0.98, 0.02] and eye=[1, 0] are comparable
        dist = kernel.cdist(prob, self.eye)
        m = th.exp(-dist)
        
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters)

class GCL(LossTerm):
    """Graph Contrastive Learning Loss (simplified without negative sampling and uncertainty filtering)"""
    large_num = 1e9

    def __init__(self, cfg):
        super().__init__()
        # Set similarity function
        if cfg.contrastive_similarity == "cos":
            self.similarity_func = self._cosine_similarity
        elif cfg.contrastive_similarity == "gauss":
            self.similarity_func = kernel.vector_kernel
        else:
            raise RuntimeError(f"Invalid contrastive similarity: {cfg.contrastive_similarity}")

    @staticmethod
    def _norm(mat):
        return th.nn.functional.normalize(mat, p=2, dim=1)

    @classmethod
    def _normalized_projections(cls, net):
        n_views = len(net.backbone_outputs)
        n = net.projections.size(0) // n_views
        # 按照视图数量切分投影特征
        hs = [cls._norm(net.projections[i*n : (i+1)*n]) for i in range(n_views)]
        return n, hs

    @classmethod
    def _cosine_similarity(cls, projections):
        h = cls._norm(projections)
        return h @ h.t()

    def __call__(self, net, cfg, extra):
        # 支持多视图对比
        n_views = len(net.backbone_outputs)
        n, hs = self._normalized_projections(net)

        labels = th.arange(0, n, device=config.DEVICE, dtype=th.long)
        masks = th.eye(n, device=config.DEVICE)

        total_loss = 0
        pair_count = 0
        
        # 计算所有视图间的两两对比损失
        for i in range(n_views):
            for j in range(n_views):
                if i == j: continue
                
                h_anchor = hs[i]
                h_positive = hs[j]
                
                # 相同视图内的负样本筛选 (推开自己视图内的其他样本)
                logits_self = ((h_anchor @ h_anchor.t()) / cfg.tau) - masks * self.large_num
                # 跨视图的正样本与负样本 (推开另一视图内的其他样本)
                logits_cross = (h_anchor @ h_positive.t()) / cfg.tau
                
                loss_ij = th.nn.functional.cross_entropy(th.cat((logits_cross, logits_self), dim=1), labels)
                total_loss += loss_ij
                pair_count += 1

        return total_loss / pair_count


class UCL(GCL):
    """
    Uncertainty-weighted Contrastive Loss (Upgrade version of GCL)
    1. Removes hard-coded uncertainty thresholds.
    2. Uses most confident samples as negatives based on ratio.
    3. Uses Dynamic Temperature: tau_i = tau_base + alpha * u_i
    """

    def _draw_confident_negative_samples(self, net, cfg, v, pos_indices):
        batch_size = net.evidence_fused.size(0)
        n_total = v * batch_size
        device = net.evidence_fused.device

        # net.u is (v*B,) already from CUE
        u_total = net.u.detach()

        # Vectorized view-exclusion mask: (num_anchors, n_total)
        # mask[i, j] is True if anchor i and sample j belong to the same original GRAHP sample
        # (regardless of which augmented view it is)
        sample_indices_anchors = pos_indices % batch_size
        sample_indices_all = th.arange(n_total, device=device) % batch_size
        
        same_sample_mask = (sample_indices_anchors.view(-1, 1) == sample_indices_all.view(1, -1))

        # Number of negative samples per anchor
        n_neg = int(cfg.negative_samples_ratio * n_total)
        if n_neg < 1: n_neg = 1

        # Use uncertainty for ranking: lower is better
        # Mask out all views of the same sample by assigning high uncertainty (2.0)
        ranking_scores = u_total.view(1, -1).expand(pos_indices.size(0), -1).clone()
        ranking_scores[same_sample_mask] = 2.0  # Max uncertainty is 1.0, so 2.0 is infinity

        # Select top-K most confident (smallest uncertainty)
        # Since we excluded all views of the same sample, the remaining are all valid "negatives"
        k = min(n_neg, n_total - v) # Max possible negatives is total - views_per_sample
        k = max(1, k)
        _, selected_indices = th.topk(ranking_scores, k=k, largest=False, dim=1)

        return selected_indices

    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        device = logits.device
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length, device=device)
            _lower_inds = th.arange(i * n, v * n, device=device)
            if DEBUG_MODE:
                assert _upper.size() == _lower.size() == _upper_inds.size() == _lower_inds.size() == (diag_length,)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds

    def _loss_with_negative_sampling(self, net, cfg, extra):
        n = net.evidence_fused.size(0)
        v = len(net.backbone_outputs)
        device = net.evidence_fused.device

        # Apply Dynamic Temperature: tau_i = tau_base + alpha * u_i
        u = net.u.detach().to(device)
        dynamic_tau = cfg.tau + cfg.tau_alpha * u # (vB,)

        # Similarity matrix
        logits = self.similarity_func(net.projections) / dynamic_tau.view(-1, 1)  # (vB, vB)

        # Get positive diagonals and their indices
        pos, pos_inds = self._get_positive_samples(logits, v, n)

        # Draw confident negative samples
        neg_inds = self._draw_confident_negative_samples(net, cfg, v, pos_inds)

        # Extract negative log-probabilities
        neg_logits = logits[pos_inds.view(-1, 1), neg_inds]

        # Final cross entropy
        inputs = th.cat((pos.view(-1, 1), neg_logits), dim=1)
        labels = th.zeros(inputs.size(0), device=device, dtype=th.long)

        loss = th.nn.functional.cross_entropy(inputs, labels)
        return loss

    def __call__(self, net, cfg, extra):
        return self._loss_with_negative_sampling(net, cfg, extra)

class RC(LossTerm):
    def __call__(self, net, cfg, extra):
        device = net.evidence_fused.device
        return self.get_dc_loss(net.evidences, device)

    def get_dc_loss(self, evidences, device):
        num_views = len(evidences)
        batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
        p = torch.zeros((num_views, batch_size, num_classes)).to(device)
        u = torch.zeros((num_views, batch_size)).to(device)
        for v in range(num_views):
            alpha = evidences[v] + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            p[v] = alpha / S
            u[v] = torch.squeeze(num_classes / S)
        dc_sum = 0
        for i in range(num_views):
            pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
            cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
            dc = pd * cc
            dc_sum = dc_sum + torch.sum(dc, dim=0)
        dc_sum = torch.mean(dc_sum)
        return dc_sum


def hidden_kernel(net, cfg):

    return kernel.vector_kernel(net.hidden, cfg.rel_sigma)

class Loss(nn.Module):
    TERM_CLASSES = {
        "GC1": GC1,
        "GC2": GC2,
        "GC3": GC3,
        "GCL": GCL,
        "UCL": UCL,
        "RC": RC,
    }
    EXTRA_FUNCS = {
        "hidden_kernel": hidden_kernel,
    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]
        self.terms = []
        for term_name in self.names:
            self.terms.append(self.TERM_CLASSES[term_name](cfg))
        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))
        

    def forward(self, net, ignore_in_total=tuple()):
        extra = {name: self.EXTRA_FUNCS[name](net, self.cfg) for name in self.required_extras_names}
        loss_values = {}
        
                # 计算原始损失值
        raw_losses = []
        for i, (name, term) in enumerate(zip(self.names, self.terms)):
            value = term(net, self.cfg, extra)
            if isinstance(value, dict):
                # 如果返回字典，只取第一个值（通常是主损失）
                value = list(value.values())[0]
            raw_losses.append(value)
        
        # 固定权重模式
        for i, (name, loss_value, weight) in enumerate(zip(self.names, raw_losses, self.weights)):
            loss_values[name] = weight * loss_value
        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])

        return loss_values