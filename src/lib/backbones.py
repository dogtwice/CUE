import torch.nn as nn
from .graph_backbones import GIN


class Backbones(nn.Module):
    BACKBONE_CONSTRUCTORS = {
        "GIN": GIN,
    }

    def __init__(self, backbone_configs, flatten_output=True):
        """
        Class representing multiple backbones. Call with list of inputs, where inputs[0] goes into the first backbone,
        and so on.

        :param backbone_configs: List of backbone configs. Each element corresponds to a backbone.
        :param flatten_output: Flatten the backbone outputs?
        :type flatten_output: bool
        """
        super().__init__()

        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            self.backbones.append(self.create_backbone(cfg, flatten_output=flatten_output))

    @property
    def output_sizes(self):
        return [bb.output_size for bb in self.backbones]

    @classmethod
    def create_backbone(cls, cfg, flatten_output=True):
        if cfg.class_name not in cls.BACKBONE_CONSTRUCTORS:
            raise RuntimeError(f"Invalid backbone: '{cfg.class_name}'")
        return cls.BACKBONE_CONSTRUCTORS[cfg.class_name](cfg, flatten_output=flatten_output)

    def forward(self, views):
        # 共享 Backbone 模式: 如果只有一个 backbone 但有多个视图，则循环调用该 backbone
        if len(self.backbones) == 1 and len(views) > 1:
            outputs = [self.backbones[0](v) for v in views]
        else:
            assert len(views) == len(self.backbones), f"n_views ({len(views)}) != n_backbones ({len(self.backbones)})."
            outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        return outputs
