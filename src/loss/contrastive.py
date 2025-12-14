from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from mmengine.dist import all_gather, get_rank
from mmpretrain.registry import MODELS
from torch import nn

# Copyright (c) OpenMMLab. All rights reserved.


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature
        self.loss_module = MODELS.build(dict(type="CrossEntropyLoss"))

    @staticmethod
    def _create_buffer(
        batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mask and the index of positive samples.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device of backend.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The mask for feature selection.
            - The index of positive samples.
            - The mask of negative samples.
        """
        mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).to(device)
        pos_idx = (
            torch.arange(batch_size * 2).to(device),
            2
            * torch.arange(batch_size, dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 2)
            .view(-1, 1)
            .squeeze()
            .to(device),
        )
        neg_mask = torch.ones(
            (batch_size * 2, batch_size * 2 - 1), dtype=torch.uint8
        ).to(device)
        neg_mask[pos_idx] = 0
        return mask, pos_idx, neg_mask

    def loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,), dtype=torch.long).to(pos.device)

        loss = self.loss_module(logits, labels)
        return loss

    def forward(self, embeddings: torch.Tensor, return_dict=True, **batch):
        """
        embeddings : tensor
            A tensor of shape (2 * B, D)
        """

        z = embeddings / (torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_idx, neg_mask = self._create_buffer(N, s.device)

        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_idx].unsqueeze(1)  # (2N)x1

        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        loss = self.loss(positive, negative)
        if return_dict:
            losses = dict(loss=loss)
            return losses
        return loss


class ContrastiveProbingLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()

        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        **batch
    ):
        return {
            "loss": F.cross_entropy(logits, labels)
            + self.contrastive_loss(embeddings, return_dict=False)
        }
