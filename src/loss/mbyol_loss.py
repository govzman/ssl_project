import torch
import torch.nn.functional as F
from torch import nn

from src.loss.pixel_reconstruction_loss import PixelReconstructionLoss


class MBYOLLoss(nn.Module):
    """
    Combined Loss for M-BYOL.
    Combines Masked Autoencoder reconstruction loss (MSE) and BYOL distillation loss (Cosine Sim).

    Equation:
        L_total = (1 - lambda_byol) * L_mae + lambda_byol * L_byol

    Balancing logic:
        - L_byol is usually in range [0.5, 3.0] (approx 2.0 at start).
        - L_mae is usually in range [0.02, 0.06].
        - To balance gradients, we assign a small weight to BYOL or large to MAE.
        - Default lambda_byol=2e-2 implies:
          0.98 * 0.04 (MAE) + 0.02 * 2.0 (BYOL) ≈ 0.04 + 0.04.
          So both terms contribute equally.
    """

    def __init__(self, mim_criterion: str = "L2", lambda_byol: float = 0.02):
        super().__init__()

        self.lambda_byol = lambda_byol
        self.lambda_mae = 1.0 - lambda_byol

        self.mae_loss_fn = PixelReconstructionLoss(criterion=mim_criterion)

    def byol_loss_func(self, p, z):
        """
        Negative Cosine Similarity.
        Args:
            p: Predictor output from Online Network
            z: Projector output from Target Network
        """
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        # 2 - 2 * cos(theta). Range: [0, 4]. Optimal: 0.
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def forward(
        self,
        pred_pixel,
        target_pixel,
        mask,  # MAE outputs
        online_pred,
        target_proj,  # BYOL outputs
        **batch
    ):
        """
        Args structure matches the output dict of MBYOL.forward()
        """

        # MAE Loss calculation
        # The PixelReconstructionLoss handles masking internally
        mae_loss_dict = self.mae_loss_fn(
            pred=pred_pixel, target=target_pixel, mask=mask
        )
        l_mae = mae_loss_dict["loss"]

        # BYOL Loss calculation
        l_byol = self.byol_loss_func(online_pred, target_proj)

        # Weighted Sum
        loss = (self.lambda_mae * l_mae) + (self.lambda_byol * l_byol)

        return {
            "mae_loss": l_mae,
            "byol_loss": l_byol,
            "loss": loss,  # Main optimization target
        }


class MBYOLProbingLoss(nn.Module):
    """
    Wrapper for M-BYOL Loss that includes Classification Loss for online linear probing.
    """

    def __init__(self, mim_criterion: str = "L2", lambda_byol: float = 0.02):
        super().__init__()
        self.mbyol_loss = MBYOLLoss(mim_criterion, lambda_byol)

    def forward(
        self,
        # Probing inputs
        logits,
        labels,
        # MBYOL inputs
        pred_pixel,
        target_pixel,
        mask,
        online_pred,
        target_proj,
        **batch
    ):
        # Calc base pretraining losses
        losses = self.mbyol_loss(
            pred_pixel=pred_pixel,
            target_pixel=target_pixel,
            mask=mask,
            online_pred=online_pred,
            target_proj=target_proj,
        )

        # Calc classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Update output dict
        losses["cls_loss"] = cls_loss
        losses["loss"] = losses["loss"] + cls_loss

        return losses
