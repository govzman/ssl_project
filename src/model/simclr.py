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
        with_probe=False,
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
        self.with_probe = with_probe
        if self.with_probe:
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

        representations_i = self.backbone(images)
        representations_j = self.backbone(images)  # tuple with tensor (B, C)
        embeddings_i = self.head(representations_i)[0]
        embeddings_j = self.head(representations_j)[0]  # tensor (B, out_channels)
        outputs = dict(embeddings_i=embeddings_i, embeddings_j=embeddings_j)
        if self.with_probe:
            outputs["logits"] = self.probe(embeddings_i.detach())
        return outputs
