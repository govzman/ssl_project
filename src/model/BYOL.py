from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.logging import MessageHub
from mmengine.utils import digit_version
from mmpretrain.models.selfsup.base import BaseSelfSupervisor
from mmpretrain.models.utils import CosineEMA
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample

if digit_version(torch.__version__) >= digit_version("1.10.0"):
    torch_meshgrid = partial(torch.meshgrid, indexing="ij")
else:
    torch_meshgrid = torch.meshgrid


def build_2d_sincos_position_embedding(
    patches_resolution: Union[int, Sequence[int]],
    embed_dims: int,
    temperature: Optional[int] = 10000.0,
    cls_token: Optional[bool] = False,
) -> torch.Tensor:
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
    assert embed_dims % 4 == 0, "Embed dimension must be divisible by 4."
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


class SSL_BYOL(nn.Module):
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
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        base_momentum=0.996,
        max_epochs=100,
        steps_per_epoch=1000,
    ) -> None:
        super().__init__()
        norm_cfg = dict(type="BN1d")
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
                norm_cfg=norm_cfg,
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
                norm_cfg=norm_cfg,
                # with_bias=False,
                # with_last_bn=False,
                # with_last_bias=False,
            )
        )

        self.backbone.init_weights()
        self.init_pos_embed()

        # momentum model
        message_hub = MessageHub.get_current_instance()
        max_iters = max_epochs * steps_per_epoch
        message_hub.update_info("max_iters", max_iters)

        self.target_backbone = CosineEMA(self.backbone, momentum=base_momentum)
        self.target_projector = CosineEMA(self.projector, momentum=base_momentum)

    def init_pos_embed(self) -> None:
        pos_embed = build_2d_sincos_position_embedding(
            patches_resolution=self.backbone.patch_resolution,
            embed_dims=self.backbone.embed_dims,
            cls_token=True,
        )

        del self.backbone.pos_embed
        self.backbone.register_buffer("pos_embed", pos_embed)

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
        online_pred1 = self.predictor((online_proj1,))[0]  # (B, 256)

        online_rep2 = self.backbone(images2)
        online_proj2 = self.projector(online_rep2)[0]
        online_pred2 = self.predictor((online_proj2,))[0]

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
