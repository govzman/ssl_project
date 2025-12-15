from mmpretrain.registry import MODELS
from torch import nn
from torch.nn import Sequential


class SSLModel(nn.Module):
    """
    Wrapper for fine-tuning, linear probe etc
    """

    def __init__(
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        in_features=768,
        out_features=10,
        freeze_backbone=True,
    ):
        """
        Args:
            backbone (nn.Module): backbone model.
            in_features (int): number of in features
            out_features (int): number of out features
        """
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
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, images, **batch):
        """
        Model forward method.

        Args:
            images (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.head(self.backbone(images)["logits"])}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
