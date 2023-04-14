import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from net.aspp import build_aspp
from net.decoder import build_decoder
from net.backbone import build_backbone
from net.reconstruction_decoder import ReconstructionDecoderFromEmbeddings 
from net.coupling_decoder import CouplingEmbSegmReconGlobalOnlyDecoderFullRes, CouplingEmbSegmReconGlobalOnlyDecoderFullResInpaintWide
from net.local_reconstruction_embedding import LocalPatchEmbedding


class DeepLab(nn.Module):
    def __init__(self, cfg, num_classes):
        super(DeepLab, self).__init__()

        output_stride = cfg.MODEL.OUT_STRIDE
        if cfg.MODEL.BACKBONE == 'drn':
            output_stride = 8

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.aspp = build_aspp(cfg.MODEL.BACKBONE, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, cfg.MODEL.BACKBONE, BatchNorm)

        self.freeze_bn = cfg.MODEL.FREEZE_BN 

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class DeepLabLocalCommon(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(DeepLabLocalCommon, self).__init__()
        self.freeze_bn = cfg.MODEL.FREEZE_BN 
        
        # inicialize trained segmentation model
        self.deeplab = DeepLab(cfg, cfg.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS)
        if not os.path.isfile(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL):
            raise RuntimeError("=> pretrained segmentation model not found at '{}'" .format(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL))
        checkpoint = torch.load(cfg.MODEL.RECONSTRUCTION.SEGM_MODEL, map_location="cpu")
        self.deeplab.load_state_dict(checkpoint['state_dict'])
        for parameter in self.deeplab.parameters():
            parameter.requires_grad = False
        del checkpoint
        
        # Local patch embeddings 
        self.local_embedding = LocalPatchEmbedding(cfg)

    def forward(self, input):
        return None 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_10x_lr_params(self):
        return None        
    
    def get_1x_lr_params(self):
        return None


class DeepLabEmbeddingGlobalOnlySegmFullRes(DeepLabLocalCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabEmbeddingGlobalOnlySegmFullRes, self).__init__(cfg, **kwargs)
        self.use_blur_input = cfg.MODEL.RECONSTRUCTION.BLUR_IMG
        self.loss_margin = cfg.LOSS.EMBEDDING_MARGIN
        
        self.blur_transform = transforms.GaussianBlur(11.0, sigma=2.0) #(7.0, 1.5))

        self.global_dec = ReconstructionDecoderFromEmbeddings(cfg)
        self.coupeling_net = CouplingEmbSegmReconGlobalOnlyDecoderFullRes(cfg)

    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            # [B, C, Hs, Ws]
            segmentation_logits = self.deeplab.decoder(x, low_level_feat)

        # [B, E, Hf, Wf]
        emb = self.local_embedding(input, encoder_feat, low_level_feat)
        
        # blur image for decoders and loss
        if self.use_blur_input:
            input_blur = self.blur_transform(input)
        else:
            input_blur = input

        # [B, 3, Hr, Wr], [B, 1, Hr, Wr]
        recon_img, recon_loss = self.global_dec(input_blur, emb, encoder_feat)

        coupling, extras = self.coupeling_net(input, emb, segmentation_logits, recon_loss)

        anomaly_score = F.softmax(coupling, dim=1)[:, 0:1, ...]

        return {
                "input": input,
                "input_blur": input_blur,
                "segmentation": F.interpolate(segmentation_logits, size=input.size()[2:], mode='bilinear', align_corners=True),
                "embeddings": emb,
                "binary_segmentation": coupling,
                "recon_img": recon_img,
                "recon_loss": recon_loss,
                "drivable_segm_mask": extras["segm_road_mask"],
                "drivable_global_recon_mask": extras["global_recon_mask"],
                "emb_dist_channel": extras["emb_dist_channel"],
                "segm_channel": segmentation_logits,
                "anomaly_score": anomaly_score,
               }


class DeepLabEmbeddingGlobalOnlySegmFullResInpaint_wide(DeepLabLocalCommon):
    def __init__(self, cfg, **kwargs):
        super(DeepLabEmbeddingGlobalOnlySegmFullResInpaint_wide, self).__init__(cfg, **kwargs)
        self.use_blur_input = cfg.MODEL.RECONSTRUCTION.BLUR_IMG
        self.loss_margin = cfg.LOSS.EMBEDDING_MARGIN
        
        self.blur_transform = transforms.GaussianBlur(11.0, sigma=2.0) #(7.0, 1.5))

        self.global_dec = ReconstructionDecoderFromEmbeddings(cfg)
        self.coupeling_net = CouplingEmbSegmReconGlobalOnlyDecoderFullResInpaintWide(cfg)

    def forward(self, input):
        with torch.no_grad():
            encoder_feat, low_level_feat = self.deeplab.backbone(input)
            x = self.deeplab.aspp(encoder_feat)
            # [B, C, Hs, Ws]
            segmentation_logits = self.deeplab.decoder(x, low_level_feat)

        # [B, E, Hf, Wf]
        emb = self.local_embedding(input, encoder_feat, low_level_feat)
        
        # blur image for decoders and loss
        if self.use_blur_input:
            input_blur = self.blur_transform(input)
        else:
            input_blur = input

        # [B, 3, Hr, Wr], [B, 1, Hr, Wr]
        recon_img, recon_loss = self.global_dec(input_blur, emb, encoder_feat)

        coupling, extras = self.coupeling_net(input, emb, segmentation_logits, recon_loss)
        anomaly_score = F.softmax(coupling, dim=1)[:, 0:1, ...]

        return {
                "input": input,
                "input_blur": input_blur,
                "segmentation": F.interpolate(segmentation_logits, size=input.size()[2:], mode='bilinear', align_corners=True),
                "embeddings": emb,
                "binary_segmentation": coupling,
                "recon_img": recon_img,
                "recon_loss": recon_loss,
                "drivable_segm_mask": extras["segm_road_mask"],
                "drivable_global_recon_mask": extras["global_recon_mask"],
                "emb_dist_channel": extras["emb_dist_channel"],
                "segm_channel": segmentation_logits,
                "anomaly_score": anomaly_score,
                "inpainting": extras["inpainting"], 
                "inpainting_mask": extras["inpainting_mask"],
                "inpainting_mask_res": extras["inpainting_mask_res"],
                "perceptual_loss": extras["perceptual_loss"], 
               }

