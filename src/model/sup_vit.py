from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing import List, Optional, Tuple, Union

import torch

import torch.nn as nn

from mmpretrain.registry import MODELS


class SupViT(nn.Module):
    def __init__(self, backbone_arch='small', img_size=96, patch_size=8, drop_rate=0.0, 
                drop_path_rate=0.0, num_classes=10):
        super().__init__()

        self.backbone = MODELS.build(
            dict(
                type="VisionTransformer",
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                out_type="cls_token",
                final_norm=True,
                with_cls_token=True,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        )

        # if embed_dim is None:
        embed_dim = self.backbone.embed_dims
        
        self.cls_head = nn.Linear(embed_dim, num_classes)


    def forward(self, images, **batch):
        feats = self.backbone(images)   # (B, C)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        logits = self.cls_head(feats)
        return dict(logits=logits)
  
