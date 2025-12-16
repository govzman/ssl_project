from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing import List, Optional, Tuple, Union
from mmengine.utils import digit_version

import torch

import torch.nn as nn

from mmpretrain.registry import MODELS

if digit_version(torch.__version__) >= digit_version('1.10.0'):
    torch_meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    torch_meshgrid = torch.meshgrid

def build_2d_sincos_position_embedding(
        patches_resolution: Union[int, Sequence[int]],
        embed_dims: int,
        temperature: Optional[int] = 10000.,
        cls_token: Optional[bool] = False) -> torch.Tensor:
    """The function is to build position embedding for model to obtain the
    position information of the image patches.

    Args:
        patches_resolution (Union[int, Sequence[int]]): The resolution of each
            patch.
        embed_dims (int): The dimension of the embedding vector.
        temperature (int, optional): The temperature parameter. Defaults to
            10000.
        cls_token (bool, optional): Whether to concatenate class token.
            Defaults to False.

    Returns:
        torch.Tensor: The position embedding vector.
    """

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch_meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


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
        self.backbone.init_weights()
        self.init_pos_embed()

        # if embed_dim is None:
        embed_dim = self.backbone.embed_dims
        
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def init_pos_embed(self) -> None:

        pos_embed = build_2d_sincos_position_embedding(
            patches_resolution=self.backbone.patch_resolution,
            embed_dims=self.backbone.embed_dims,
            cls_token=True,
        )

        del self.backbone.pos_embed
        self.backbone.register_buffer('pos_embed', pos_embed)


    def forward(self, images, **batch):
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        logits = self.cls_head(feats)
        return dict(logits=logits)
  
