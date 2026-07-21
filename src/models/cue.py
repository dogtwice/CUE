import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import helpers
from ..lib.augmentation import graph_augment
from ..lib.backbones import Backbones
from ..lib.evidence import UDF, EvidenceHead
from ..lib.loss import Loss
from ..lib.optimizer import Optimizer
from .model_base import ModelBase


class CUE(ModelBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden = None
        self.backbone_outputs = None
        self.projections = None

        self.backbones = Backbones(cfg.backbone_configs)
        num_classes = cfg.cm_config.n_clusters
        num_views = cfg.fusion_config.n_views
        self.udf = UDF(num_views, num_classes)

        bb_sizes = self.backbones.output_sizes
        assert all(bb_sizes[0] == s for s in bb_sizes), f"CUE requires all backbones to have the same output size. Got: {bb_sizes}"

        self.projector = nn.Identity()
        backbone_output_dim = bb_sizes[0][0] if isinstance(bb_sizes[0], list) else bb_sizes[0]
        self.evidence_head = EvidenceHead(input_dim=[backbone_output_dim], cfg=cfg.cm_config)
        self.loss = Loss(cfg=cfg.loss_config)

        self.backbones.apply(helpers.he_init_weights)
        self.evidence_weights = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

    def init_centers_with_kmeans(self, loader, device):
        self.train()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        all_features = []
        with torch.no_grad():
            for _, views in enumerate(loader):
                if isinstance(views, (list, tuple)) and len(views) == 3:
                    maybe_views = views[0]
                    view0 = maybe_views[0] if isinstance(maybe_views, (list, tuple)) else maybe_views
                else:
                    view0 = views[0] if isinstance(views, (list, tuple)) else views
                if hasattr(view0, "to"):
                    view0 = view0.to(device)
                v_orig = self.backbones([view0])[0]
                hidden = self.evidence_head.hidden(v_orig)
                all_features.append(hidden.detach().cpu())

        if not all_features:
            print("[Init] Warning: No features collected.")
            return

        all_features = torch.cat(all_features, dim=0)
        self.evidence_head.perform_spectral_init(all_features, random_state=42)

    def forward(self, views):
        n_views = self.cfg.fusion_config.n_views
        if len(self.backbones.backbones) == 1:
            if not isinstance(views, (list, tuple)):
                views = [views] * n_views
            elif len(views) < n_views:
                views = list(views) + [views[0]] * (n_views - len(views))

        if self.training and len(self.backbones.backbones) == 1 and len(views) == n_views and all(v is views[0] for v in views):
            if hasattr(views[0], "edge_index"):
                aug_cfg = self.cfg.augmentation_config
                new_views = [views[0]]
                for view_idx in range(1, n_views):
                    if aug_cfg is not None:
                        new_views.append(
                            graph_augment(
                                views[0],
                                node_drop_rate=aug_cfg.node_drop_rate,
                                edge_drop_rate=aug_cfg.edge_drop_rate,
                                feat_mask_rate=aug_cfg.feat_mask_rate,
                            )
                        )
                    else:
                        new_views.append(graph_augment(views[0]))
                views = new_views

        self.backbone_outputs = self.backbones(views)
        self.projections = self.projector(torch.cat(self.backbone_outputs, dim=0))
        self.evidences = {}
        self.hiddens = {}

        num_classes = self.cfg.cm_config.n_clusters
        for v in range(n_views):
            self.evidences[v], self.hiddens[v] = self.evidence_head(self.backbone_outputs[v])

        self.hidden = self.average_tensors(self.hiddens)
        self.evidences = self.apply_evidence_weights(self.evidences)
        self.evidence_fused, self.predict, self.u_fused = self.udf(self.backbone_outputs, self.evidences)
        self.u = torch.cat(n_views * [self.u_fused], dim=0).view(-1)
        return self.evidence_fused

    def apply_evidence_weights(self, evidences):
        weights = F.softmax(self.evidence_weights, dim=0)
        return {v: evidences[v] * weights[v] for v in evidences.keys()}

    def average_tensors(self, tensor_dict):
        sum_tensor = None
        for tensor in tensor_dict.values():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("All values must be Tensor")
            sum_tensor = tensor.clone() if sum_tensor is None else sum_tensor + tensor
        if sum_tensor is None:
            raise ValueError("The dictionary is empty or does not contain a valid Tensor.")
        return sum_tensor / len(tensor_dict)
