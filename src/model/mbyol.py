import torch
from mmengine.logging import MessageHub
from mmpretrain.models import NonLinearNeck
from mmpretrain.models.utils import CosineEMA
from mmpretrain.registry import MODELS
from torch import nn

from src.model.mae_vit import SSL_MAEPretrainDecoder, SSL_MAEPretrainHead, SSL_MAEViT


# @MODELS.register_module()
class MBYOL(nn.Module):
    def __init__(
        self,
        backbone_arch="small",
        img_size=96,
        patch_size=8,
        mask_ratio=0.5,
        base_momentum=0.004,
        proj_dim=256,
        pred_dim=256,
        num_classes=10,
        drop_rate=0.0,
        drop_path_rate=0.0,
        with_probe=False,
        max_epochs=50,
        steps_per_epoch=98,
    ):
        super().__init__()
        norm_cfg = dict(type="BN1d")

        # Online Backbone (MAEViT)
        self.backbone = MODELS.build(
            dict(
                type="SSL_MAEViT",
                mask_ratio=mask_ratio,
                arch=backbone_arch,
                img_size=img_size,
                patch_size=patch_size,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                out_type="raw",
            )
        )

        # MAE Branch: Decoder and Head Helper
        self.vit_decoder = MODELS.build(
            dict(
                type="SSL_MAEPretrainDecoder",
                num_patches=(img_size // patch_size) ** 2,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=self.backbone.embed_dims,
                decoder_embed_dim=256,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4,
                norm_cfg=dict(type="LN", eps=1e-6),
                init_cfg=[
                    dict(type="Xavier", layer="Linear", distribution="uniform"),
                    dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
                ],
            )
        )

        self.reconstruction_head = MODELS.build(
            dict(
                type="SSL_MAEPretrainHead",
                patch_size=patch_size,
                loss=dict(type="PixelReconstructionLoss", criterion="L2"),
                norm_pix=True,
            )
        )

        # BYOL Branch: Projector & Predictor (Online)
        self.projector = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=self.backbone.embed_dims,
                hid_channels=512,
                out_channels=256,
                num_layers=2,
                with_avg_pool=False,
                norm_cfg=norm_cfg,
            )
        )

        self.predictor = MODELS.build(
            dict(
                type="NonLinearNeck",
                in_channels=proj_dim,
                hid_channels=512,
                out_channels=pred_dim,
                num_layers=2,
                with_avg_pool=False,
                with_last_bn=False,
                norm_cfg=dict(type="BN1d"),
            )
        )

        # Init Weights
        self.backbone.init_weights()
        self.vit_decoder.init_weights()
        self.projector.init_weights()
        self.predictor.init_weights()

        message_hub = MessageHub.get_current_instance()
        max_iters = max_epochs * steps_per_epoch
        message_hub.update_info("max_iters", max_iters)

        # 5. Target Network setup (using CosineEMA as requested)
        self.target_backbone = CosineEMA(self.backbone, momentum=base_momentum)
        self.target_projector = CosineEMA(self.projector, momentum=base_momentum)

    def forward(self, images: torch.Tensor, images2: torch.Tensor, **batch):
        """
        images  (View A): Сильная аугментация -> Маскирование -> MAE + BYOL Online
        images2 (View B): Слабая аугментация (или стандартная) -> Full View -> BYOL Target
        """
        self.target_backbone.update_parameters(self.backbone)
        self.target_projector.update_parameters(self.projector)

        # --- Online Branch (Masked) ---
        # x_online: [B, N_masked+1, Dim] (0-й токен = CLS)
        x_online, mask, ids_restore = self.backbone(images, mask=True)

        # Split для задач
        cls_online = x_online[:, 0]  # Take [CLS] token
        latent_patches = (
            x_online  # Keep all including CLS for decoder (implementation specific)
        )

        # MAE Task: Predict Pixels
        # reconstruction: [B, N_patches, PixelDim]
        pred_pixels = self.vit_decoder(latent_patches, ids_restore)
        target_pixels = self.reconstruction_head.construct_target(images)

        # BYOL Task: Predict Target Representation
        proj_online = self.projector((cls_online,))[0]
        pred_online = self.predictor((proj_online,))[0]

        # --- Target Branch (Full - No Mask) ---
        with torch.no_grad():
            # Важно: вызываем forward target backbone без маски (mask=False)
            # x_target: [B, N_all+1, Dim]
            feat_target = self.target_backbone(images2, mask=False)

            if isinstance(feat_target, (list, tuple)):
                feat_target = feat_target[0]

            # Берем глобальный CLS токен от полного изображения
            cls_target = feat_target[:, 0]

            # Проецируем
            proj_target = self.target_projector((cls_target,))[0]
            proj_target = proj_target.detach()

        # --- Outputs Collection ---
        outputs = dict(
            # MAE components
            pred_pixel=pred_pixels,
            target_pixel=target_pixels,
            mask=mask,
            # BYOL components
            online_pred=pred_online,
            target_proj=proj_target,
        )

        return outputs
