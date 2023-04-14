import os
import torch
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import copy
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
from net.loss import reshape_embeddings_and_labels

class TensorboardSummary(object):
    def __init__(self, cfg, directory):
        self.directory = directory
        self.current_cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
        self.current_cmap.set_bad(color='black')
        self.dump_dir_tr = os.path.join(self.directory, "vis", "train")
        self.dump_dir_val = os.path.join(self.directory, "vis", "val")
        os.makedirs(self.dump_dir_tr, exist_ok=True)
        os.makedirs(self.dump_dir_val, exist_ok=True)
        self.img_mean = np.array(cfg.INPUT.NORM_MEAN)[None, None, :]
        self.img_std = np.array(cfg.INPUT.NORM_STD)[None, None, :]
        self.model = cfg.MODEL.NET

    def denormalize_img(self, img):
        return img*self.img_std + self.img_mean
    def denormalize_img_patch(self, img):
        return img*self.img_std.transpose(2, 0, 1) + self.img_mean.transpose(2, 0, 1)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    
    def decode_target(self, segm):
        labels = [0, 1]
        colors = [[255, 0, 0],[128,64,128]]
        new_segm = 255*np.ones(shape=list(segm.shape)+[3], dtype=np.uint8)
        for l in labels:
            new_segm[segm==l, :] = colors[l] 
        return new_segm

    def visualize_image(self, writer, dataset, image, target, output, global_step, epoch, epoch_id, validation=False):
        if isinstance(output, dict):
            self.visualize_dict_switch(writer, dataset, image, target, output, global_step, epoch, epoch_id, validation)
        else:
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Predicted label', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image('Groundtruth label', grid_image, global_step)

    def visualize_dict_switch(self, writer, dataset, image, target, output, global_step, epoch, epoch_id, validation=False):
        if self.model == "DeepLabEmbeddingGlobalOnlySegmFullResMultiClass":
            return 
        elif self.model[:16] == "DeepLabEmbedding":
            self.visualize_emb(writer, dataset, image, target, output, global_step, epoch, epoch_id, validation)
        else:
            filename = os.path.join(self.dump_dir_val if validation else self.dump_dir_tr, "e{:04d}_i{:08d}.jpg".format(epoch, epoch_id))
            grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(output["anomaly_score"][:3].clone().cpu().data, 3, normalize=False, range=(0, 1))
            writer.add_image('Anomaly score', grid_image, global_step)
            
            segm_map = (255*decode_seg_map_sequence(torch.max(output["segmentation"], 1)[1].detach().cpu().numpy(), dataset=dataset).numpy()).astype(np.uint8)
            segm_masked = output.get("segmentation_masked", None)
            second_segm = 0 
            if segm_masked is not None:
                second_segm = 1 
                segm_map2 = (255*decode_seg_map_sequence(torch.max(segm_masked, 1)[1].detach().cpu().numpy(), dataset=dataset).numpy()).astype(np.uint8)

            target_map = self.decode_target(target.detach().cpu().numpy())
            w = image.size()[3]
            h = image.size()[2]
            b = image.size()[0]
            out_img = np.zeros(shape=[6*h + second_segm*h, b*w, 3], dtype=np.uint8)
            for i in range(0, b):
                out_img[:h, i*w:(i+1)*w, :] = (255*self.denormalize_img(image[i, ...].clone().detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)
                err_img = (255*self.current_cmap(output["anomaly_score"][i, 0, ...].clone().detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
                out_img[h:2*h, i*w:(i+1)*w, :] = err_img
                out_img[2*h:3*h, i*w:(i+1)*w, :] = (255*output["recon_img"][i, ...].clone().detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
                err_img = (255*self.current_cmap(output["recon_loss"][i, 0, ...].clone().detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
                out_img[3*h:4*h, i*w:(i+1)*w, :] = err_img
                out_img[4*h:5*h, i*w:(i+1)*w, :] = segm_map[i, ...].transpose(1, 2, 0)
                out_img[5*h:6*h, i*w:(i+1)*w, :] = target_map[i, ...]
                if second_segm > 0:
                    out_img[6*h:7*h, i*w:(i+1)*w, :] = segm_map2[i, ...].transpose(1, 2, 0)

            out_img = cv2.resize(out_img, (int(out_img.shape[1]/2), int(out_img.shape[0]/2)))
            cv2.imwrite(filename, out_img[:, :, ::-1])


    def visualize_emb(self, writer, dataset, image, target, output, global_step, epoch, epoch_id, validation=False):
        filename = os.path.join(self.dump_dir_val if validation else self.dump_dir_tr, "e{:04d}_i{:08d}.jpg".format(epoch, epoch_id))
        # plot 2D projection using PCA
        emb, labels = reshape_embeddings_and_labels(output["embeddings"].detach(), target)
        _, _, V = torch.pca_lowrank(emb, center=True)
        proj = torch.matmul(emb, V[:, :2]).cpu().numpy() 
        labels = labels.cpu().numpy()
        
        fig = plt.figure(42)
        fig.clf()
        mask = (labels == 1)
        plt.scatter(proj[mask, 0], proj[mask, 1], c="g", label="Road")
        mask = (labels == 0)
        plt.scatter(proj[mask, 0], proj[mask, 1], c="r", label="Anomaly")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(prop={'size': 14,'weight': "medium" }, loc="upper right")
        fig.savefig(filename[:-3] + "png", dpi=250, bbox_inches='tight')   
        
        if "recon_patches" in output and "input_patches" in output:
            # [B, Hf*Wf, 3, Hr, Wr], [B, Hf*Wf, 3, Hr, Wr]
            recon_patches, input_patches = output["recon_patches"].detach().cpu().numpy(), output["input_patches"].detach().cpu().numpy()
            recon_loss_patches = np.sqrt(np.mean(np.power((recon_patches - input_patches), 2), axis=(2, 3, 4)) + 1e-9)
            hp, wp = input_patches.shape[-2:]
            b = image.size()[0]
            emb_size = output["embeddings"].size()[-2:]
            img_patch = np.zeros(shape=[b, 3, hp*emb_size[0], wp*emb_size[1]])
            recon_patch = np.zeros(shape=[b, 3, hp*emb_size[0], wp*emb_size[1]])
            recon_loss_patch = np.zeros(shape=[b, 3, hp*emb_size[0], wp*emb_size[1]])
            for i in range(0, b):
                for r in range(0, emb_size[0]):
                    for c in range(0, emb_size[1]):
                        idd = r*emb_size[1] + c
                        img_patch[i, :, r*hp:(r+1)*hp, c*wp:(c+1)*wp] = (input_patches[i, idd, ...])*255
                        img_patch[i, :, r*hp:(r+1)*hp, c*wp] = np.zeros(shape=[3,1])
                        img_patch[i, :, r*hp, c*wp:(c+1)*wp] = np.zeros(shape=[3,1]) 
                        
                        recon_patch[i, :, r*hp:(r+1)*hp, c*wp:(c+1)*wp] = (recon_patches[i, idd, ...])*255 
                        recon_patch[i, :, r*hp:(r+1)*hp, c*wp] = np.zeros(shape=[3,1])
                        recon_patch[i, :, r*hp, c*wp:(c+1)*wp] = np.zeros(shape=[3,1]) 

                        recon_loss_patch[i, :, r*hp:(r+1)*hp, c*wp:(c+1)*wp] = 255*np.array(self.current_cmap(recon_loss_patches[i, idd]))[:3, None, None]
                        recon_loss_patch[i, :, r*hp:(r+1)*hp, c*wp] = np.zeros(shape=[3,1])
                        recon_loss_patch[i, :, r*hp, c*wp:(c+1)*wp] = np.zeros(shape=[3,1]) 

            grid = img_patch.transpose(0, 2, 3, 1).astype(np.uint8)
            grid_r = recon_patch.transpose(0, 2, 3, 1).astype(np.uint8)
            grid_rl = recon_loss_patch.transpose(0, 2, 3, 1).astype(np.uint8)
            h, w = grid.shape[1:3]
            out_img = np.zeros(shape=[b*h, 3*w, 3], dtype=np.uint8)
            for i in range(0, b):
                out_img[i*h:(i+1)*h, :w, :] = grid[i, ...] 
                out_img[i*h:(i+1)*h, w:2*w, :] = grid_r[i, ...] 
                out_img[i*h:(i+1)*h, 2*w:3*w, :] = grid_rl[i, ...] 
            out_img = cv2.resize(out_img.astype(np.uint8), (int(out_img.shape[1]/2), int(out_img.shape[0]/2)))
            cv2.imwrite(filename[:-4] + "_grid_patches.png", out_img[:, :, ::-1])
        
        if "emb_dist_channel" in output:
            w = image.size()[3]
            h = image.size()[2]
            b = image.size()[0]
            hf, wf = output["embeddings"].size()[2:] 

            out_img = np.zeros(shape=[b*h, 7*w, 3], dtype=np.uint8)
            for i in range(0, b):
                emb_dist = 255*self.current_cmap(output["emb_dist_channel"].detach().cpu().numpy()[:, 0, ...])[:, :, :, :3]
                segm_mask = 255*(output["drivable_segm_mask"].detach().cpu().numpy().reshape(-1, hf, wf) > 0)
                if "drivable_local_recon_mask" in output: 
                    local_mask = 255*(output["drivable_local_recon_mask"].detach().cpu().numpy().reshape(-1, hf, wf) > 0)
                if "drivable_global_recon_mask" in output:
                    global_mask = 255*(output["drivable_global_recon_mask"].detach().cpu().numpy().reshape(-1, hf, wf) > 0)

                out_img[i*h:(i+1)*h, :w, :] = (255*self.denormalize_img(image[i, ...].detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)
                out_img[i*h:(i+1)*h, w:2*w, :] = cv2.resize(
                        255*self.current_cmap(output["emb_dist_channel"][i, 0, ...].detach().cpu().numpy())[:, :, :3],
                        (w, h), interpolation=cv2.INTER_NEAREST)
                out_img[i*h:(i+1)*h, 2*w:3*w, :] = cv2.resize(segm_mask[i, ...].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[:, :, None]
                
                if output["emb_dist_channel"].size(1) > 2:
                    out_img[i*h:(i+1)*h, 3*w:4*w, :] = cv2.resize(
                            255*self.current_cmap(output["emb_dist_channel"][i, 2, ...].detach().cpu().numpy())[:, :, :3],
                            (w, h), interpolation=cv2.INTER_NEAREST)

                if "drivable_local_recon_mask" in output: 
                    out_img[i*h:(i+1)*h, 4*w:5*w, :] = cv2.resize(local_mask[i, ...].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[:, :, None]

                out_img[i*h:(i+1)*h, 5*w:6*w, :] = cv2.resize(
                        255*self.current_cmap(output["emb_dist_channel"][i, 1, ...].detach().cpu().numpy())[:, :, :3],
                        (w, h), interpolation=cv2.INTER_NEAREST)
                if "drivable_global_recon_mask" in output:
                    out_img[i*h:(i+1)*h, 6*w:7*w, :] = cv2.resize(global_mask[i, ...].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[:, :, None]

            cv2.imwrite(filename[:-4] + "_emb_clustering.png", out_img[:, :, ::-1])
        
        if "anomaly_score" in output:
            w = image.size()[3]
            h = image.size()[2]
            b = image.size()[0]
            out_img = np.zeros(shape=[b*h, 2*w, 3], dtype=np.uint8)
            for i in range(0, b):
                out_img[i*h:(i+1)*h, :w, :] = (255*self.denormalize_img(image[i, ...].detach().cpu().numpy().transpose(1, 2, 0)))
                out_img[i*h:(i+1)*h, w:2*w, :] = 255*self.current_cmap(output["anomaly_score"][i, 0, ...].detach().cpu().numpy())[:, :, :3]
            cv2.imwrite(filename[:-4] + "_final.png", out_img[:, :, ::-1])

        if "recon_img" in output:
            w = image.size()[3]
            h = image.size()[2]
            b = image.size()[0]
            add = 0
            if "input_blur" in output:
                add = 1
            out_img = np.zeros(shape=[b*h, (add+3)*w, 3], dtype=np.uint8)
            for i in range(0, b):
                out_img[i*h:(i+1)*h, :w, :] = (255*self.denormalize_img(image[i, ...].detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8)

                out_img[i*h:(i+1)*h, w:2*w, :] = cv2.resize((255*output["recon_img"][i, ...].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8), (w, h))

                err_img = (255*self.current_cmap(output["recon_loss"][i, 0, ...].detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
                out_img[i*h:(i+1)*h, 2*w:3*w, :] = cv2.resize(err_img, (w, h))

                if "input_blur" in output:
                     out_img[i*h:(i+1)*h, 3*w:4*w, :] = (255*self.denormalize_img((output["input_blur"])[i, ...].detach().cpu().numpy().transpose(1, 2, 0))).astype(np.uint8) 
            out_img = cv2.resize(out_img, (int(out_img.shape[1]/2), int(out_img.shape[0]/2)))
            cv2.imwrite(filename[:-4] + "_global.png", out_img[:, :, ::-1])




