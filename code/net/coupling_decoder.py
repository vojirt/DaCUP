import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kornia.morphology import dilation

from net.aspp import ASPP
from net.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from net.deepfillv2_singleton import create_generator, PerceptualPixelLossTrainableMerge, random_ff_mask


class CouplingEmbSegmReconGlobalOnlyDecoderFullRes(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CouplingEmbSegmReconGlobalOnlyDecoderFullRes, self).__init__()
        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.net = nn.Sequential(
                   ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=12, _inplanes=22),
                   ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=6, _inplanes=12),
                   nn.Conv2d(6, 2, kernel_size=1, stride=1, padding=0),
                   nn.BatchNorm2d(2),
                   nn.ReLU(inplace=True)
                )
        
        self.global_recon_loss_thr = 0.2

    def forward(self, input, emb, segmentation_logits, recon_loss):
        # input: [B, 3, H, W]
        # emb: [B, E, Hf, Wf]
        # segmentation logits: [B, C, Hs, Ws]
        # recon_loss: [B, 1, H, W]

        size_HW = input.size()[2:]

        # k-means like test for road embeddings
        # [B, Hf, Wf]
        segm_labels = torch.argmax(F.interpolate(segmentation_logits, size=emb.size()[2:], mode='bilinear', align_corners=True), dim=1)
        # drivable classes mask
        # [B, Hf * Wf]
        segm_road_mask = ((segm_labels == 0) | (segm_labels == 1)).flatten(start_dim=1)
        global_recon_mask = (F.interpolate(recon_loss, size=emb.size()[2:], mode='bilinear', align_corners=True) < self.global_recon_loss_thr).flatten(start_dim=1)
        # [B, 2, E]
        mean_road_emb = torch.zeros(size=(emb.size(0), 2, emb.size(1)), device=emb.get_device())
        for b in range(0, emb.size(0)):
            # segm emb mean
            if torch.sum(segm_road_mask[b, ...]) > 0:
                mean_road_emb[b, 0, :] = emb[b, ...].flatten(start_dim=1).transpose(0, 1)[segm_road_mask[b, ...], :].mean(dim=0)
            # global recon mean
            if torch.sum(global_recon_mask[b, ...]) > 0:
                mean_road_emb[b, 1, :] = emb[b, ...].flatten(start_dim=1).transpose(0, 1)[global_recon_mask[b, ...], :].mean(dim=0)

        # [B, 2, Hf, Wf]
        emb_dist = ((emb[:, None, :, :, :]-mean_road_emb[:, :, :, None, None]).pow(2).sum(dim=2) + 1e-9).sqrt()
        
        # [B, 2, Hs, Ws]
        emb_dist_channel  = F.interpolate(emb_dist, size=size_HW, mode='bilinear', align_corners=True)

        # [B, 3+segm_logits(19), Hs, Ws]
        concat_channels = torch.cat([
                    F.interpolate(segmentation_logits, size=size_HW, mode='bilinear', align_corners=True), 
                    emb_dist_channel, 
                    recon_loss,
                ],dim=1)
        coupling = self.net(concat_channels)
        
        return coupling, {
                "segm_road_mask": segm_road_mask,
                "global_recon_mask": global_recon_mask,
                "emb_dist_channel": emb_dist_channel,
                }


class CouplingEmbSegmReconGlobalOnlyDecoderFullResInpaintWide(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CouplingEmbSegmReconGlobalOnlyDecoderFullResInpaintWide, self).__init__()
        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

        self.net = nn.Sequential(
                   ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=12, _inplanes=23),
                   ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=6, _inplanes=12),
                   nn.Conv2d(6, 2, kernel_size=1, stride=1, padding=0),
                   nn.ReLU(inplace=True)
                )

        self.inpaint_net = create_generator(cfg.MODEL.INPAINT_WEIGHTS_FILE).eval()
        for parameter in self.inpaint_net.parameters():
            parameter.requires_grad = False

        self.perceptual_loss = PerceptualPixelLossTrainableMerge()
        
        self.global_recon_loss_thr = torch.nn.Parameter(torch.ones(1, device="cuda", dtype=torch.float) * 0.1, requires_grad=True)

    def forward(self, input, emb, segmentation_logits, recon_loss):
        # input: [B, 3, H, W]
        # emb: [B, E, Hf, Wf]
        # segmentation logits: [B, C, Hs, Ws]
        # recon_loss: [B, 1, H, W]

        if input.is_cuda and self.mean_tensor.get_device() != input.get_device():
            self.mean_tensor = self.mean_tensor.cuda(input.get_device())
            self.std_tensor = self.std_tensor.cuda(input.get_device())

        size_HW = input.size()[2:]
        global_recon_loss_thr = torch.clamp(0.05+self.global_recon_loss_thr.sigmoid(), 0.0, 0.95)

        # k-means like test for road embeddings
        # [B, Hf, Wf]
        segm_labels = torch.argmax(F.interpolate(segmentation_logits, size=emb.size()[2:], mode='bilinear', align_corners=True), dim=1)
        # drivable classes mask
        # [B, Hf * Wf]
        segm_road_mask = ((segm_labels == 0) | (segm_labels == 1)).flatten(start_dim=1)
        global_recon_mask = (F.interpolate(recon_loss, size=emb.size()[2:], mode='bilinear', align_corners=True) < global_recon_loss_thr).flatten(start_dim=1)
        # [B, 2, E]
        mean_road_emb = torch.zeros(size=(emb.size(0), 2, emb.size(1)), device=emb.get_device())
        for b in range(0, emb.size(0)):
            # segm emb mean
            if torch.sum(segm_road_mask[b, ...]) > 0:
                mean_road_emb[b, 0, :] = emb[b, ...].flatten(start_dim=1).transpose(0, 1)[segm_road_mask[b, ...], :].mean(dim=0)
            # global recon mean
            if torch.sum(global_recon_mask[b, ...]) > 0:
                mean_road_emb[b, 1, :] = emb[b, ...].flatten(start_dim=1).transpose(0, 1)[global_recon_mask[b, ...], :].mean(dim=0)

        # [B, 2, Hf, Wf]
        emb_dist = ((emb[:, None, :, :, :]-mean_road_emb[:, :, :, None, None]).pow(2).sum(dim=2) + 1e-9).sqrt()
        
        # [B, 2, Hs, Ws]
        emb_dist_channel  = F.interpolate(emb_dist, size=size_HW, mode='bilinear', align_corners=True)

        inpaint_mask = (recon_loss > global_recon_loss_thr).float()
        img_sz_area = input.size(2)*input.size(3)

        if self.training:
            # add random mask stroke
            # set the same free form masks for each batch
            mask_random = torch.zeros_like(inpaint_mask)
            for i in range(input.size(0)):
                mask_random[i, 0, ...] = torch.tensor(random_ff_mask(shape=inpaint_mask.size()[2:]).astype(np.float32), 
                                             device=inpaint_mask.get_device(), requires_grad=False)
            inpaint_mask = torch.logical_or(inpaint_mask > 0, mask_random > 0).float()

        input_range = torch.clamp(input*self.std_tensor+self.mean_tensor, 0, 1)
        if img_sz_area > 1280*720:
            new_size_factor = np.sqrt(1280*720 / img_sz_area)
            new_sz = np.array(new_size_factor * np.array(input.size()[2:]), dtype=int)
            # Must be divisible by 8 for the otherwise the GAN crashes (strides and resizing issues, prob) 
            new_sz = np.round((new_sz/2**3))*(2**3)         
            new_sz = tuple(new_sz.astype(int).tolist())
            input_res = F.interpolate(input_range, mode='bilinear', size=new_sz, align_corners=False)
            inpaint_mask_res = (F.interpolate(inpaint_mask, mode='bilinear', size=new_sz, align_corners=False) > 0).float()
        else:
            input_res = input_range
            inpaint_mask_res = inpaint_mask

        inpaint_mask_res = (dilation(inpaint_mask_res, torch.ones(7, 7, device=input.get_device())) > 0).float()
        with torch.no_grad():
            _, second_out = self.inpaint_net(input_res, inpaint_mask_res)
        
        # forward propagation
        input_inpainted = input_res * (1 - inpaint_mask_res) + second_out * inpaint_mask_res

        perceptual_loss = self.perceptual_loss(input_res, input_inpainted)
        perceptual_loss = F.interpolate(perceptual_loss, mode='bilinear', size=input.size()[2:], align_corners=False)

        if img_sz_area > 1280*720:
            input_inpainted  = F.interpolate(input_inpainted, mode='bilinear', size=input.size()[2:], align_corners=False)

        # [B, 19+2 + 1 + 1, Hs, Ws]
        concat_channels = torch.cat([
                    F.interpolate(segmentation_logits, size=size_HW, mode='bilinear', align_corners=True), 
                    emb_dist_channel, 
                    recon_loss,
                    perceptual_loss
                ],dim=1)
        coupling = self.net(concat_channels)
        
        return coupling, {
                "segm_road_mask": segm_road_mask,
                "global_recon_mask": global_recon_mask,
                "emb_dist_channel": emb_dist_channel,
                "inpainting": input_inpainted,
                "inpainting_mask": inpaint_mask,
                "inpainting_mask_res": inpaint_mask_res, 
                "perceptual_loss": perceptual_loss,
                }

