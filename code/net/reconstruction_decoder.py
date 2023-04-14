import torch
from torch import nn
from torch.nn import functional as F
from net.loss import SSIMLoss


class ReconstructionDecoderFromEmbeddings(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ReconstructionDecoderFromEmbeddings, self).__init__()
        self.layers_dim = [16, 32, 64, 128]
        #self.embedding_part_dim = int(0.75*cfg.MODEL.LOCAL_RECONSTRUCTION.EMBEDDING_DIM)
        self.embedding_part_dim = cfg.MODEL.LOCAL_RECONSTRUCTION.EMBEDDING_DIM
        self.skip_conn_flag = cfg.MODEL.RECONSTRUCTION.SKIP_CONN

        self.mean_tensor = torch.FloatTensor(cfg.INPUT.NORM_MEAN)[None, :, None, None]
        self.std_tensor = torch.FloatTensor(cfg.INPUT.NORM_STD)[None, :, None, None]

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_size = lambda x, sz: F.interpolate(x, size=sz, mode='bilinear', align_corners=True)


        if self.skip_conn_flag:
            self.skip_conn_layer = conv_block(2048, self.embedding_part_dim)
            self.dec_layer0 = conv_block(2*self.embedding_part_dim, self.layers_dim[3])
        else:
            self.dec_layer0 = conv_block(self.embedding_part_dim, self.layers_dim[3])

        self.dec_layer1 = conv_block(self.layers_dim[3], self.layers_dim[2])
        self.dec_layer2 = conv_block(self.layers_dim[2], self.layers_dim[1])
        self.dec_layer3 = conv_block(self.layers_dim[1], self.layers_dim[0])
        self.final_layer = nn.Conv2d(self.layers_dim[0], out_channels=3, kernel_size=1, stride=1, padding=0)

        if cfg.MODEL.RECONSTRUCTION.ERROR_FNC == "ssim":
            self.recon_loss = SSIMLoss(window_size=11, absval=True) 
        else:
            raise NotImplementedError

    def forward(self, img, emb, encoder_feat):
        # img : [B, 3, H, W]
        # emb : [B, E, Hf, Wf]
        # encoder_feat : [B, C, Hf, Wf]

        # decoder
        if self.skip_conn_flag:
            skip = self.skip_conn_layer(encoder_feat)
            d_l0 = self.dec_layer0(self.up2(torch.cat([emb[:,:self.embedding_part_dim, ...], skip],dim=1)))
        else:
            d_l0 = self.dec_layer0(self.up2(emb[:,:self.embedding_part_dim, ...]))
        d_l1 = self.dec_layer1(self.up2(d_l0))
        d_l2 = self.dec_layer2(self.up2(d_l1))
        d_l3 = self.dec_layer3(self.up_size(d_l2, img.shape[2:]))
        
        if img.is_cuda and self.mean_tensor.get_device() != img.get_device():
            self.mean_tensor = self.mean_tensor.cuda(img.get_device())
            self.std_tensor = self.std_tensor.cuda(img.get_device())

        recons = torch.clamp(self.final_layer(d_l3)*self.std_tensor+self.mean_tensor, 0, 1)
        #recons = torch.clamp(self.final_layer(d_l3), 0, 1)
        recons_loss = self.recon_loss(recons, torch.clamp(img*self.std_tensor+self.mean_tensor, 0, 1))[:, None, ...]
        
        return [recons, recons_loss]


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    # NOTE: GeLU instead of ReLU?
    return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
           )

