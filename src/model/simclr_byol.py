import torch
from mmengine.logging import MessageHub
from mmpretrain.models import NonLinearNeck, VisionTransformer
from mmpretrain.models.utils import CosineEMA
from mmpretrain.registry import MODELS
from torch import nn

from src.model.mae_vit import build_2d_sincos_position_embedding


class SimCLR_Byol(nn.Module):
    def __init__(
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        num_classes=10,
        emb_dim=512,
        with_probe=False,
        max_epochs=100,
        steps_per_epoch=1000,
        base_momentum=0.004,
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
        self.simclr_head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        # Projector: online + target
        self.projector = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        # Predictor: online
        self.predictor = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=emb_dim,
                hid_channels=512,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        # momentum model
        message_hub = MessageHub.get_current_instance()
        max_iters = max_epochs * steps_per_epoch
        message_hub.update_info("max_iters", max_iters)

        self.target_backbone = CosineEMA(self.backbone, momentum=base_momentum)
        self.target_projector = CosineEMA(self.projector, momentum=base_momentum)

        self.with_probe = with_probe
        if self.with_probe:
            self.probe = nn.Linear(768, num_classes)

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
        # Online network forward pass
        representations_i = self.backbone(images)  # (B, 768)
        online_proj1 = self.projector(representations_i)[0]  # (B, out_channels)
        online_pred1 = self.predictor((online_proj1,))[0]  # (B, out_channels)

        representations_j = self.backbone(images2)
        online_proj2 = self.projector(representations_j)[0]
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

        embeddings_i = self.simclr_head(representations_i)[0]
        embeddings_j = self.simclr_head(representations_j)[
            0
        ]  # tensor (B, out_channels)

        outputs = dict(
            embeddings_i=embeddings_i,
            embeddings_j=embeddings_j,
            online_pred1=online_pred1,
            online_pred2=online_pred2,
            target_proj1=target_proj1,
            target_proj2=target_proj2,
        )

        if self.with_probe:
            outputs["logits"] = self.probe(representations_i[0].detach())
        return outputs
