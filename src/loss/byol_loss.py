import torch
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from torch import nn


class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss between predictions and targets.

        Args:
            pred (torch.Tensor): Predictions from online network, shape (B, D)
            target (torch.Tensor): Projections from target network, shape (B, D)

        Returns:
            torch.Tensor: Loss value
        """
        # Normalize predictions and targets
        pred = F.normalize(pred, dim=1, p=2)
        target = F.normalize(target, dim=1, p=2)

        # Compute cosine similarity and convert to loss (negative cosine similarity)
        loss = 2 - 2 * (pred * target).sum(dim=1).mean()
        return loss

    def forward(
        self,
        online_pred1: torch.Tensor,
        online_pred2: torch.Tensor,
        target_proj1: torch.Tensor,
        target_proj2: torch.Tensor,
        return_dict=True,
        **batch
    ):
        """
        Compute BYOL loss.

        Args:
            online_pred1: Online network predictions for view 1, shape (B, D)
            online_pred2: Online network predictions for view 2, shape (B, D)
            target_proj1: Target network projections for view 1, shape (B, D)
            target_proj2: Target network projections for view 2, shape (B, D)
        """

        # BYOL loss: predict target projection of view 2 from online view 1, and vice versa
        loss_1 = self.regression_loss(online_pred1, target_proj2)
        loss_2 = self.regression_loss(online_pred2, target_proj1)

        # Symmetrized loss
        loss = loss_1 + loss_2
<<<<<<< HEAD

        losses = dict(loss=loss)
        return losses
=======
        if return_dict:
            losses = dict(loss=loss)
            return losses
        return loss
>>>>>>> 36a8b1a (SimCLR + byol)
