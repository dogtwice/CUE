import math
from typing import Any, Optional

import torch
import torch.nn as nn


def replace_gin_post_bn_with_gn(net: nn.Module, gn_groups: int = 8) -> int:
    """Replace only the post-conv BatchNorm layers in GIN backbones with GroupNorm.

    This targets `GIN.bns` (one BN per GNN layer), and intentionally does NOT touch:
    - the BN inside each GINConv's MLP (`conv.nn`)
    - any BN inside the evidence head

    Returns the number of replaced layers.
    """

    replaced = 0

    backbones = getattr(getattr(net, "backbones", None), "backbones", None)
    if backbones is None:
        return 0

    for bb in backbones:
        if bb.__class__.__name__ != "GIN":
            continue

        gin = bb
        bns = getattr(gin, "bns", None)
        if bns is None:
            continue

        for idx, bn in enumerate(bns):
            if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue

            num_features = int(bn.num_features)
            groups = min(int(gn_groups), num_features)
            groups = math.gcd(num_features, groups)
            groups = max(1, groups)

            new_layer = nn.GroupNorm(num_groups=groups, num_channels=num_features, affine=True)

            weight = getattr(bn, "weight", None)
            if weight is not None:
                new_layer = new_layer.to(device=weight.device, dtype=weight.dtype)

            gin.bns[idx] = new_layer
            replaced += 1

    return replaced


def apply_gn_backbone_post(net: nn.Module, gn_groups: int = 8, verbose: bool = False) -> int:
    """Apply `gn_backbone_post` normalization variant to the given network."""

    n_replaced = replace_gin_post_bn_with_gn(net, gn_groups=gn_groups)
    if verbose:
        print(f"[Norm] Applied gn_backbone_post (replaced {n_replaced} layers)")
    setattr(net, "_gn_backbone_post_applied", True)
    return n_replaced
