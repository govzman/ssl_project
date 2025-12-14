from torch import nn
from torch.nn import Sequential
from torchvision.models import resnet18


class ResNetModel(nn.Module):
    """
    ResNet
    """

    def __init__(self):
        """
        Args:
            depth (int): number of depth.
        """
        super().__init__()

        self.net = resnet18()
        self.net.conv1 = Sequential(
            nn.AdaptiveAvgPool2d((64, 64)), nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        )
        self.net.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, images, **batch):
        """
        Model forward method.

        Args:
            images (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(images)}

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
