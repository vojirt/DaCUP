from torch import nn
from net.aspp import ASPP


class LocalPatchEmbedding(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LocalPatchEmbedding, self).__init__()
        self.latent_dim_per_scale = cfg.MODEL.LOCAL_RECONSTRUCTION.LATENT_DIM_PER_SCALE
        self.embedding_dim = cfg.MODEL.LOCAL_RECONSTRUCTION.EMBEDDING_DIM
        self.fc_dropout_p = 0.1

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.bottleneck = ASPP(cfg.MODEL.BACKBONE, cfg.MODEL.OUT_STRIDE, BatchNorm, outplanes=self.latent_dim_per_scale)

        self.embedding_net = nn.Sequential(
                    nn.Linear(self.latent_dim_per_scale*5, 2*self.latent_dim_per_scale),   # 5 scales in ASPP each with latent_dim_per_scale channels
                    nn.ReLU(),
                    nn.Dropout(p=self.fc_dropout_p,inplace=True),
                    nn.Linear(2*self.latent_dim_per_scale, self.latent_dim_per_scale),
                    nn.ReLU(),
                    nn.Dropout(p=self.fc_dropout_p,inplace=True),
                    nn.Linear(self.latent_dim_per_scale, self.embedding_dim),
                    nn.ReLU()
                )

    def forward(self, img, encoder_feat, low_level_feat):
        # [B, C, H, W]
        bl = self.bottleneck(encoder_feat, return_concat=True)

        # reshape bl for FC layers: [B, C, H, W] -> [B, H*W, C]
        bl_reshaped = bl.flatten(start_dim=2, end_dim=-1).transpose(1, 2)
        
        # [B, H*W, E]
        emb = self.embedding_net(bl_reshaped)
        
        bl_sz = bl.size()
        # [B, E, H, W]
        return emb.transpose(1,2).reshape((bl_sz[0], self.embedding_dim, bl_sz[2], bl_sz[3]))
