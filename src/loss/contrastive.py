import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()

        self.temperature = temperature

    def forward(self, embeddings_i, embeddings_j, return_dict=True, **batch):
        B, _ = embeddings_i.shape

        z = torch.cat([embeddings_i, embeddings_j], dim=0)
        z = F.normalize(z, dim=1)

        sim_matrix = torch.mm(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-float("inf"))

        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(
            embeddings_i.device
        )

        if return_dict:
            return {"loss": F.cross_entropy(sim_matrix, labels)}
        return F.cross_entropy(sim_matrix, labels)


class ContrastiveProbingLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()

        self.contrastive_loss = ContrastiveLoss(temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings_i: torch.Tensor,
        embeddings_j: torch.Tensor,
        **batch
    ):
        # embeddings = torch.cat([embeddings_i, embeddings_j], dim=0)
        return {
            "loss": F.cross_entropy(logits, labels)
            + self.contrastive_loss(embeddings_i, embeddings_j, return_dict=False)
        }
