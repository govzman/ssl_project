import torch
from mmpretrain.registry import MODELS
from torch import nn


class SimCLR(nn.Module):
    def __init__(
        self, backbone_arch="small", img_size=96, patch_size=8, num_classes=10
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
        out_channels = 512
        self.head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=out_channels,
                num_layers=2,
                with_avg_pool=False,
            )
        )
        self.probe = nn.Linear(out_channels, num_classes)

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
