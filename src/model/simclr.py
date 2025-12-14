import torch
from torch import nn
from mmpretrain.registry import MODELS
    

class SimCLR(nn.Module):
    def __init__(self, backbone_arch='small', img_size=224, patch_size=16):
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
                in_channels=2048,
                hid_channels=2048,
                out_channels=128,
                num_layers=2,
                with_avg_pool=True,
            )
        )

    def forward(self, x: torch.Tensor, **batch):
        """
        x : tensor
            tensor of shape (B, C, H, W)
        """
        representations = self.backbone(x) # tuple with tensor (B, C)
        z = self.head(representations)[0] # tensor (B, out_channels)
        return representations[0]
        