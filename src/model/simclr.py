import torch
from torch import nn
from mmpretrain.registry import MODELS
    

class SimCLR(nn.Module):
    def __init__(self, backbone_arch='small', img_size=96, patch_size=8):
        super().__init__()
        self.backbone = MODELS.build(
            dict(
                type="VisionTransformer",
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                out_type="cls_token"
            )
        )

        self.head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=128,
                num_layers=2,
                with_avg_pool=False,
            )
        )

    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):
        """
        images : tensor
            tensor of shape (B, C, H, W)
        """
        
        x = torch.cat([images, images2], dim=0)
        representations = self.backbone(x) # tuple with tensor (B, C)
        z = self.head(representations)[0] # tensor (B, out_channels)
        return dict(
            embeddings=z
        )
        