from src.model.mae_vit import build_2d_sincos_position_embedding, SSL_MAEViT, SSL_MAEPretrainDecoder, SSL_MAEPretrainHead
from src.model.simclr import SimCLR

from src.loss.contrastive import ContrastiveLoss
from src.loss

import torch
from mmpretrain.models import NonLinearNeck, VisionTransformer
from mmpretrain.registry import MODELS
from torch import nn


class ContrastiveMAE(nn.Module):
    def __init__(
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        drop_rate=0.0,
        drop_path_rate=0.0,
        mask_ratio=0.5,
        num_classes=10,
        emb_dim=128,
        with_probe=False,
    ):
        super().__init__()
        self.backbone = MODELS.build(
            dict(
                type="SSL_MAEViT",
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                out_type="raw",
            )
        )
        
        self.vit_decoder = MODELS.build(
            dict(
                type='SSL_MAEPretrainDecoder',
                num_patches=(img_size // patch_size) ** 2,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=self.backbone.embed_dims,
                decoder_embed_dim=256,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4,
                norm_cfg=dict(type='LN', eps=1e-6),
                init_cfg=[
                    dict(type='Xavier', layer='Linear', distribution='uniform'),
                    dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
                ],
                predict_feature_dim=3
            )
        )

        sefl.reconstruction_head = MODELS.build(
            dict(
                type='SSL_MAEPretrainHead',
                patch_size=patch_size
            )
        )

        self.contrastive_head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=768,
                hid_channels=768,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        self.backbone.init_weights()
        self.init_pos_embed()

        self.with_probe = with_probe
        if self.with_probe:
            self.probe = nn.Linear(emb_dim, num_classes)

    def init_pos_embed(self):
        self.backbone.pos_embed.requires_grad = False
        num_patches = self.backbone.pos_embed.shape[1] - self.backbone.num_extra_tokens
        pos_embed = build_2d_sincos_position_embedding(
            int(num_patches**0.5), self.backbone.pos_embed.shape[-1], cls_token=True
        )

        self.backbone.pos_embed.data.copy_(pos_embed.float())

        w = self.backbone.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.backbone.cls_token, std=0.02)

    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):

        x1, mask1, ids_restore1 = self.backbone(images, mask=True)
        x2, mask2, ids_restore2 = self.backbone(images2, mask=True)
        reconstr1 = self.vit_decoder(x1, ids_restore1)
        reconstr2 = self.vit_decoder(x2, ids_restore2)
        target1 = self.reconstruction_head.construct_target(reconstr1, images, mask1) 
        

        representations_i = self.backbone(images)
        representations_j = self.backbone(images2)  # tuple with tensor (B, C)
        embeddings_i = self.head(representations_i)[0]
        embeddings_j = self.head(representations_j)[0]  # tensor (B, out_channels)
        outputs = dict(embeddings_i=embeddings_i, embeddings_j=embeddings_j)
        if self.with_probe:
            outputs["logits"] = self.probe(embeddings_i.detach())
        return outputs
