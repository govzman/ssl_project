from src.model.mae_vit import build_2d_sincos_position_embedding, SSL_MAEViT, SSL_MAEPretrainDecoder, SSL_MAEPretrainHead
from src.model.simclr import SimCLR

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
                mask_ratio=mask_ratio,
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
                # predict_feature_dim=3
            )
        )

        self.reconstruction_head = MODELS.build(
            dict(
                type='SSL_MAEPretrainHead',
                patch_size=patch_size,
                loss=dict( # redundant
                    type='PixelReconstructionLoss',
                    criterion='L2',
                ),
            )
        )

        self.contrastive_head = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=self.backbone.embed_dims,
                hid_channels=self.backbone.embed_dims,
                out_channels=emb_dim,
                num_layers=2,
                with_avg_pool=False,
            )
        )

        self.backbone.init_weights()
        self.vit_decoder.init_weights()
        self.reconstruction_head.init_weights()

        self.with_probe = with_probe
        if self.with_probe:
            self.probe = nn.Linear(self.backbone.embed_dims, num_classes)

   
    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):

        x1, mask1, ids_restore1 = self.backbone(images, mask=True)
        x2, mask2, ids_restore2 = self.backbone(images2, mask=True)
        reconstr1 = self.vit_decoder(x1, ids_restore1)
        reconstr2 = self.vit_decoder(x2, ids_restore2)
        target1 = self.reconstruction_head.construct_target(images)
        target2 = self.reconstruction_head.construct_target(images2)

        representations_i = x1[:, 1:, :].mean(dim=1) # avg pool of backbone outputs
        representations_j = x2[:, 1:, :].mean(dim=1)

        embeddings_i = self.contrastive_head((representations_i.unsqueeze(-1).unsqueeze(-1),))[0]
        embeddings_j = self.contrastive_head((representations_j.unsqueeze(-1).unsqueeze(-1),))[0] 

        outputs = dict(embeddings_i=embeddings_i, embeddings_j=embeddings_j,
                        pred1=reconstr1, target1=target1, mask1=mask1,
                        pred2=reconstr2, target2=target2, mask2=mask2)

        if self.with_probe:
            outputs["logits"] = self.probe(x1[:, 0, :].detach())
        return outputs
