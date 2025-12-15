import torch
from mmpretrain.models import NonLinearNeck, VisionTransformer
from mmpretrain.registry import MODELS
from torch import nn

from src.model.mae_vit import build_2d_sincos_position_embedding


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        num_classes=10,
        emb_dim=128,
    ):
        super().__init__()
        self.backbone = MODELS.build(
            dict(
                type="VisionTransformer",
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                out_type="cls_token",
            )
        )
        self.backbone.init_weights()
        self.init_pos_embed()
        self.head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )
        self.probe = nn.Linear(emb_dim, num_classes)

    def init_pos_embed(self):
        self.backbone.pos_embed.requires_grad = False
        num_patches = self.backbone.pos_embed.shape[1] - self.backbone.num_extra_tokens
        pos_embed = build_2d_sincos_position_embedding(
            int(num_patches**0.5), self.backbone.pos_embed.shape[-1], cls_token=True
        )

        self.backbone.pos_embed.data.copy_(pos_embed.float())

        w = self.backbone.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.backbone.cls_token, std=0.02)

    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):
        """
        images : tensor
            tensor of shape (B, C, H, W)
        """

        B, _, _, _ = images.shape

        x = torch.cat([images, images2], dim=0)
        representations = self.backbone(x)  # tuple with tensor (B, C)
        embeddings = self.head(representations)[0]  # tensor (B, out_channels)
        logits = self.probe(embeddings[:B].detach())
        return dict(embeddings=embeddings, logits=logits)
