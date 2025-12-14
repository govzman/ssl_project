# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.utils import CosineEMA
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class BYOL(BaseSelfSupervisor):
    """BYOL.

    Implementation of `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features
            to compact feature vectors.
        head (dict): Config dict for module of head functions.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.004.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(
        self, backbone_arch="small", img_size=96, patch_size=8, base_momentum=1
    ) -> None:
        super().__init__()

        # Online
        self.backbone = MODELS.build(
            dict(
                type="VisionTransformer",
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                out_type="cls_token",
            )
        )
        # Projector: online + target
        self.projector = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=256,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        # Predictor: online
        self.predictor = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=256,
                hid_channels=512,
                out_channels=256,
                num_layers=2,
                with_avg_pool=False,
                with_bias=False,
                with_last_bn=False,
                with_last_bias=False,
                with_last_relu=False,
            )
        )
        # momentum model
        self.target_backbone = CosineEMA(self.backbone, momentum=base_momentum)
        self.target_projector = CosineEMA(self.projector, momentum=base_momentum)

    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):
        """
        images : tensor
            tensor of shape (B, C, H, W)
        images2 : tensor
            augmented version of images, shape (B, C, H, W)
        """

        # Online network forward pass
        online_rep1 = self.backbone(images)  # (B, 768)
        online_proj1 = self.projector(online_rep1)[0]  # (B, 256)
        online_pred1 = self.predictor(online_proj1)[0]  # (B, 256)

        online_rep2 = self.backbone(images2)
        online_proj2 = self.projector(online_rep2)[0]
        online_pred2 = self.predictor(online_proj2)[0]

        # Target network forward pass (no gradients)
        with torch.no_grad():
            # Update target networks with momentum
            self.target_backbone.update_parameters(self.backbone)
            self.target_projector.update_parameters(self.projector)

            target_rep1 = self.target_backbone(images)
            target_proj1 = self.target_projector(target_rep1)[0]

            target_rep2 = self.target_backbone(images2)
            target_proj2 = self.target_projector(target_rep2)[0]

        return dict(
            online_pred1=online_pred1,
            online_pred2=online_pred2,
            target_proj1=target_proj1,
            target_proj2=target_proj2,
        )
