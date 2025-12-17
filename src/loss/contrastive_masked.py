import torch
import torch.nn.functional as F
from torch import nn

from src.loss.contrastive import ContrastiveLoss
from src.loss.pixel_reconstruction_loss import PixelReconstructionLoss

class ContrastiveMaskedLoss(nn.Module):
    def __init__(self, temperature=0.5, mim_criterion = 'L2', lambda_mim=0.5):
        super().__init__()

        self.temperature = temperature
        self.lambda_mim = lambda_mim
        self.lambda_cont = 1 - self.lambda_mim
        
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.mim_loss = PixelReconstructionLoss(criterion=mim_criterion)

    def forward(self, embeddings_i, embeddings_j, 
                pred1, target1, mask1,
                pred2, target2, mask2,
                **batch):

        contrastive_loss = self.contrastive_loss(embeddings_i, embeddings_j, return_dict=False, **batch)
        mim_loss1 = self.mim_loss(pred1, target1, mask1, **batch)['loss']
        mim_loss2 = self.mim_loss(pred2, target2, mask2, **batch)['loss']
        mim_loss = (mim_loss1 + mim_loss2) / 2

        loss = self.lambda_mim * mim_loss + self.lambda_cont * contrastive_loss

        return {'mim_loss' : mim_loss,
                'contrastive_loss' : contrastive_loss,
                'loss' : loss}

class ContrastiveMaskedProbingLoss(nn.Module):
    def __init__(self, temperature=0.5, mim_criterion = 'L2', lambda_mim=0.5):
        super().__init__()

        self.contr_mim_loss = ContrastiveMaskedLoss(temperature, mim_criterion, lambda_mim)

    def forward(self, logits, labels,
                embeddings_i, embeddings_j, 
                pred1, target1, mask1,
                pred2, target2, mask2,
                **batch):

        contr_mim_loss = self.contr_mim_loss(embeddings_i, embeddings_j, 
                                                pred1, target1, mask1,
                                                pred2, target2, mask2,
                                                **batch)

        cls_loss = F.cross_entropy(logits, labels)

        return {'loss' : contr_mim_loss + cls_loss}