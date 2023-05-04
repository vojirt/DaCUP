import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


class CrossEntropyLoss(object):
    def __init__(self, loss_cfg, weight=None, use_cuda=True, **kwargs):
        self.ignore_index = loss_cfg.IGNORE_LABEL
        self.weight = weight
        self.size_average = loss_cfg.SIZE_AVG
        self.batch_average = loss_cfg.BATCH_AVG 
        self.cuda = use_cuda

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

# modified code from https://github.com/Po-Hsun-Su/pytorch-ssim
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size, absval):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.absval = absval
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, recons, input):
        (_, channel, _, _) = input.size()
        if channel == self.channel and self.window.data.type() == input.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if input.is_cuda:
                window = window.cuda(input.get_device())
            window = window.type_as(input)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(input, self.window, padding = self.window_size//2, groups = self.channel)
        mu2 = F.conv2d(recons, self.window, padding = self.window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(input*input, self.window, padding = self.window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(recons*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(input*recons, self.window, padding = self.window_size//2, groups = self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if self.absval:
            return 1-torch.clamp(torch.abs(ssim_map), 0, 1).mean(1)
        else:
            return 1-torch.clamp(ssim_map, 0, 1).mean(1)

class ReconstructionAnomalyLoss(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL

    def __call__(self, output, target):

        inv_mask = (target == 0).float()[:, None, ...]
        mask_local = (target == 1).float()[:, None, ...]
        mask_count = (mask_local > 0).float().sum() + 1e-7
        inv_mask_count = (inv_mask > 0).float().sum() + 1e-7
        
        loss = output["recon_loss"]
        margin = 0.999
        loss_road = (F.relu(loss - (1 - margin)) * mask_local).sum() / mask_count
        loss_bg = (F.relu(margin - loss) * inv_mask).sum() / inv_mask_count  
        return loss_road + loss_bg

class TripletMarginLoss(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.margin = loss_cfg.EMBEDDING_MARGIN 
        self.dir_reg = loss_cfg.DIRECTION_TRIPLET_REGULARIZATION
        self.emb_space_reg = loss_cfg.EMB_SPACE_REG
        self.emb_space_reg_w = loss_cfg.EMB_SPACE_REG_WEIGHT
        self.epsilon = 1e-9

    def __call__(self, output, target, return_labels=False, hard_negative=False):
        emb, labels = reshape_embeddings_and_labels_opt(output["embeddings"], target)
        
        triplets = get_triplets_torch_stochastic_partiall_EPS(emb, labels, ignore_label=self.ignore_label, only_for_label=1,
                max_size_per_label=1024, hard_negative=hard_negative)

        if emb.get_device() != triplets.get_device():
            triplets = triplets.cuda(emb.get_device())

        if triplets.size()[0] > 0: 
            ap_distances = ((emb[triplets[:, 0], ...] - emb[triplets[:, 1], ...]).pow(2).sum(-1) + self.epsilon).pow(.5)
            an_distances = ((emb[triplets[:, 0], ...] - emb[triplets[:, 2], ...]).pow(2).sum(-1) + self.epsilon).pow(.5)
            if self.emb_space_reg_w > 1e-6:
                reg = self.emb_space_reg_w*(F.relu(torch.linalg.norm(emb[triplets[:, 0], ...], dim=-1) - self.emb_space_reg).mean() + 
                           F.relu(torch.linalg.norm(emb[triplets[:, 1], ...], dim=-1) - self.emb_space_reg).mean() + 
                           F.relu(torch.linalg.norm(emb[triplets[:, 2], ...], dim=-1) - self.emb_space_reg).mean())
            else:
                reg = 0.0
            loss = F.relu(ap_distances - an_distances + self.margin).mean() + reg
            if self.dir_reg > 1e-6:
                vn = emb[triplets[:, 2], ...] - emb[triplets[:, 0], ...]
                va = emb[triplets[:, 1], ...] - emb[triplets[:, 0], ...]
                loss += self.dir_reg*(((vn * va)/(torch.linalg.norm(vn) * torch.linalg.norm(va))).sum(-1)).mean()
        else:
            loss = (emb*0).sum()
        if return_labels:
            return loss, labels
        else:
            return loss 

class GlobalOnlyDecL2TripletMarginLossXentCoupeling(object):
    def __init__(self, loss_cfg, **kwargs):
        self.warm_up_dynamic_weights = loss_cfg.WARM_UP_DYNAMIC_WEIGHTS
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.xent_margin_reg = loss_cfg.XENT_MARGIN_REG
        self.epsilon = 1e-9
        self.all_pairs = None
        self.hard_negative_epoch_start = loss_cfg.HARD_NEGATIVE_START_EPOCH if loss_cfg.HARD_NEGATIVE_START_EPOCH >= 0 else 1e9
        self.triplet_loss =  TripletMarginLoss(loss_cfg)
        self.global_recon_loss = ReconstructionAnomalyLoss(loss_cfg)
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_label, reduction="mean")
        self.print_flags = [True, True, True, True]

    def __call__(self, output, target):
        # Float, [B*Hf*Wf]
        epoch = output.get("__epoch__", -1)
        hard_negative_flag = True if epoch >= self.hard_negative_epoch_start else False
        if self.print_flags[3] and hard_negative_flag:
            self.print_flags[3] = False
            print("__LOSS__: Starting to use hard negative mining for triplet loss.")

        triplet_loss, labels = self.triplet_loss(output, target, return_labels=True, hard_negative=hard_negative_flag)

        log_softmax = F.log_softmax(output["binary_segmentation"], dim=1)
        xent_loss = self.nll_loss(log_softmax, target.long())
        
        if self.xent_margin_reg: 
            softmax = F.softmax(output["binary_segmentation"], dim=1)
            xent_loss = xent_loss + F.relu(0.6 - softmax[:,0,...]).mean() + F.relu(softmax[:,1,...] - 0.4).mean()

        w_xent, w_tri, w_local_recon, w_global_recon = 0.6, 0.2, 0.2, 0.2 #0.6, 0.4, 0.2, 0.2 

        if self.warm_up_dynamic_weights and epoch > -1:
            if epoch < 5:
                w_xent, w_tri, w_local_recon, w_global_recon = 0.0, 1.2, 0.0, 0.0 
                if self.print_flags[0]:
                    self.print_flags[0] = False
                    print("__LOSS__: Using warm up dynamic weights w_xent, w_tri, w_local_recon, w_global_recon = 0.0, 1.2, 0.0, 0.0")
                if torch.isnan(xent_loss).any():
                    xent_loss = 0
            elif epoch < 10:
                w_xent, w_tri, w_local_recon, w_global_recon = 0.0, 0.4, 0.4, 0.4
                if self.print_flags[1]:
                    self.print_flags[1] = False
                    print("__LOSS__: Using warm up dynamic weights w_xent, w_tri, w_local_recon, w_global_recon = 0.0, 0.4, 0.4, 0.4")
                if torch.isnan(xent_loss).any():
                    xent_loss = 0
            else:
                if self.print_flags[2]:
                    self.print_flags[2] = False
                    print("__LOSS__: Using default weights w_xent, w_tri, w_local_recon, w_global_recon = 0.6, 0.2, 0.2, 0.2")

        return w_xent*xent_loss + w_tri*triplet_loss + w_global_recon*0.5*self.global_recon_loss(output, target)

# ------------------------------------------------------------------------------

def reshape_embeddings_and_labels_opt(embeddings, target, majority_nn_vote=True, anomaly_wins=False, return_weights=False, road_label=1):
    # embeddings : [B, E, Hf, Wf]
    # target : [B, H, W]
    
    # TODO : better aggregation of the target labels, e.g. if we want 
    #        some "uncertain" region we may want to know percentage of labels 
    #        aggregated during resizing (e.g. 75% road, 25% anomaly)

    labels_type = target.dtype
    Hf, Wf = embeddings.size()[2:]    
    H, W = target.size()[1:]
    B = target.size(0)
    if majority_nn_vote:
        h_step = int(np.ceil(H/float(Hf)))
        w_step = int(np.ceil(W/float(Wf))) 
        unique_labels = torch.unique(target)
        # [B, L, Hf, Wf]
        counts_per_label = F.interpolate(
                                nn.AvgPool2d((h_step, w_step))
                                    (
                                       torch.stack([(target == ul).float() for ul in unique_labels], dim=1)
                                    ), 
                                size=[Hf, Wf], mode="bilinear", align_corners=True) 
        # [B * Hf * Wf]
        labels = unique_labels[torch.argmax(counts_per_label, dim=1).flatten()]
        if anomaly_wins:
            idx = torch.nonzero(unique_labels == 0) 
            if idx.size(0) > 0:
                anomaly_u_idx = idx[0, 0]
                anomaly_mask = counts_per_label[:, anomaly_u_idx, ...].flatten() > 0
                labels[anomaly_mask] = 0 

        if torch.sum(unique_labels==road_label) > 0:
            weights = (counts_per_label[:, unique_labels==road_label, ...] / torch.sum(counts_per_label, dim=1)).flatten()
        else:
            weights = torch.zeros_like(labels)
    else:
        # [B * Hf * Wf]
        labels = torch.flatten(F.interpolate(target.unsqueeze(1).float(), size=[Hf, Wf], mode="nearest")).type(labels_type)
        weights = torch.ones_like(labels)
    # [B * Hf * Wf, E]
    emb = torch.flatten(embeddings.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
    if return_weights:
        return emb, labels, weights
    else:
        return emb, labels

# Levi et al., "Rethinking preventing class-collapsing in metric learning with margin-based losses", ICCV2021
def get_triplets_torch_stochastic_partiall_EPS(emb, labels, ignore_label=255, only_for_label=None, skip_labels=[], max_size_per_label=-1, hard_negative=False, return_triplets_labels=False):
    # embeddings : [B * Hf * Wf, E]
    triplets = None 
    triplets_labels = None
    label_mask_ignore = (labels == ignore_label)
    for label in torch.unique(labels):
        if (label == ignore_label) or (only_for_label is not None and label != only_for_label) or (label in skip_labels):
            continue
        label_mask = (labels == label)
        label_indices = torch.nonzero(label_mask, as_tuple=True)[0]
        if label_indices.size()[0] < 2:
            continue
        negative_indices = torch.nonzero(torch.logical_not(label_mask | label_mask_ignore), as_tuple=True)[0]
        if negative_indices.size(0) == 0:
            continue
        
        if max_size_per_label > 0 and label_indices.size()[0] > max_size_per_label:
            label_indices = label_indices[torch.randperm(label_indices.size()[0])[:max_size_per_label]]

        # Only closest anchor-positive pairs
        dist_positive = (emb[label_indices, None, :] - emb[None, label_indices, :]).pow(2).sum(-1).fill_diagonal_(1e9)
        pos_idx = torch.argmin(dist_positive, dim=1)
        ap = torch.stack([label_indices, label_indices[pos_idx]], dim=1) 

        if not hard_negative:
            # semi-hard negatives, Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
            dist_negative = (emb[ap[:, 0], None, :] - emb[None, negative_indices, :]).pow(2).sum(-1)
            pairs_dist = (emb[ap[:, 0], :] - emb[ap[:,1], :]).pow(2).sum(-1)
            dist_negative[dist_negative < pairs_dist[:, None]] = 1e9 
            neg_idx = torch.argmin(dist_negative, dim=1)
        else:
            dist_negative = (emb[ap[:, 0], None, :] - emb[None, negative_indices, :]).pow(2).sum(-1)
            neg_idx = torch.argmin(dist_negative, dim=1)

        temp_triplets = torch.cat([ap, negative_indices[neg_idx][:, None]], dim=1)
        if triplets is None:
            triplets = temp_triplets
            triplets_labels = label*torch.ones(temp_triplets.size(0), device=temp_triplets.get_device(), dtype=torch.long)
        else:
            triplets = torch.cat([triplets, temp_triplets], dim=0)
            triplets_labels = torch.cat([triplets_labels, (label*torch.ones(temp_triplets.size(0), device=temp_triplets.get_device())).long()], dim=0)

    if triplets is None:
        triplets = torch.LongTensor(np.array([]))
        triplets_labels = torch.LongTensor(np.array([]))

    if return_triplets_labels:
        return triplets, triplets_labels
    else:
        return triplets 
