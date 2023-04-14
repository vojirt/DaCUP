import os
import sys
import torch
import torch.nn.functional as F
import importlib
import numpy as np
import skimage

#from config import get_cfg_defaults
sys.path.append("<PATH TO REPO ROOT DIR>/code/config/")
from defaults import get_cfg_defaults

class MethodEvaluator():
    def __init__(self, **kwargs):
        """ Model initialization. """
        raise NotImplementedError

    def evaluate(self, image):
        """ Implement forward pass for a particular method. Return anomaly score per pixel. """
        raise NotImplementedError


class ReconAnom(MethodEvaluator):
    def __init__(self, **kwargs) -> None:
        self.exp_dir = kwargs["exp_dir"]
        self.code_dir = os.path.join(self.exp_dir, "code")
        self.print_crops_stats_once = True
        self.print_crops_stats_once_2 = True
        self.print_slic_stats_once = True

        cfg_local = get_cfg_defaults() 

        if os.path.isfile(os.path.join(self.exp_dir, "parameters.yaml")):
            with open(os.path.join(self.exp_dir, "parameters.yaml"), 'r') as f:
                cc = cfg_local._load_cfg_from_yaml_str(f)
            cfg_local.merge_from_file(os.path.join(self.exp_dir, "parameters.yaml"))
            cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
        else:
            assert False, "Experiment directory does not contain parameters.yaml: {}".format(self.exp_dir)
        if os.path.isfile(os.path.join(self.code_dir, "checkpoints", "checkpoint-best.pth")):
            cfg_local.EXPERIMENT.RESUME_CHECKPOINT = os.path.join(self.code_dir, "checkpoints", "checkpoint-best.pth")
        elif (cfg_local.EXPERIMENT.RESUME_CHECKPOINT is None 
                or not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT)):
            assert False, "Experiment dir does not contain best checkpoint, or no checkpoint specified or specified checkpoint does not exist: {}".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT)

        if not torch.cuda.is_available(): 
            print ("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        if cfg_local.MODEL.SYNC_BN is None:
            if cfg_local.SYSTEM.USE_GPU and len(cfg_local.SYSTEM.GPU_IDS) > 1:
                cfg_local.MODEL.SYNC_BN = True
            else:
                cfg_local.MODEL.SYNC_BN = False

        if cfg_local.INPUT.BATCH_SIZE_TRAIN is None:
            cfg_local.INPUT.BATCH_SIZE_TRAIN = 4 * len(cfg_local.SYSTEM.GPU_IDS)

        if cfg_local.INPUT.BATCH_SIZE_TEST is None:
            cfg_local.INPUT.BATCH_SIZE_TEST = cfg_local.INPUT.BATCH_SIZE_TRAIN

        cfg_local.freeze()
        self.device = torch.device("cuda:0" if cfg_local.SYSTEM.USE_GPU else "cpu")

        self.mean_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_MEAN)[None, :, None, None].to(self.device)
        self.std_tensor = torch.FloatTensor(cfg_local.INPUT.NORM_STD)[None, :, None, None].to(self.device)

        # Define network
        sys.path.insert(0, self.code_dir)
        kwargs = {'cfg': cfg_local}
        spec = importlib.util.spec_from_file_location("models", os.path.join(self.exp_dir, "code", "net", "models.py"))
        model_module = spec.loader.load_module()
        print (self.exp_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)
        sys.path = sys.path[1:]

        if cfg_local.EXPERIMENT.RESUME_CHECKPOINT is not None:
            if not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT))
            checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
            if cfg_local.SYSTEM.USE_GPU and torch.cuda.device_count() > 1:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
            del checkpoint
        else:
            raise RuntimeError("=> model checkpoint has to be provided for testing!")

        # Using cuda
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model.coupeling_net, "global_recon_loss_thr"):
            print (f"global_recon_loss_thr: {self.model.coupeling_net.global_recon_loss_thr}")
        if hasattr(self.model.coupeling_net, "perceptual_loss"):
            if hasattr(self.model.coupeling_net.perceptual_loss, "pyramid_weights"):
                print (f"perceptual weights: {self.model.coupeling_net.perceptual_loss.pyramid_weights}")

        to_del = []
        for k, v in sys.modules.items():
            if k[:3] == "net":
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

    def evaluate(self, image, return_full=False):
        # [1, 3, H, W]
        img = (image.to(self.device) - self.mean_tensor)/self.std_tensor
        assert img.size(0) == 1, "ReconAnom->self.evaluate: Only batch size = 1 is supported!"

        with torch.no_grad():
            output = self.model(img)
        anomaly_score = output["anomaly_score"][:, 0, ...]

        if return_full:
            return output
        else:
            return anomaly_score 


def decode_seg_map_sequence(label_masks, dataset):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'cityscapes' or dataset == "cityscapes_2class" or dataset== "citybdd100k_2class":
        n_classes = 19
        label_colours = np.array([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]])
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def main():
    from PIL import Image
    import cv2
    from matplotlib import pyplot as plt
    from matplotlib import cm
    import torchvision
    import copy

    params = {"exp_dir": "<PATH TO REPO ROOT DIR>"}
    img_pil = np.array(Image.open("<PATH TO IMAGE>"))
    out_dir = "./out/"
    os.makedirs(out_dir, exist_ok=True)

    evaluator = ReconAnom(**params)
    # [batch=1, 3, H, W]
    img = torchvision.transforms.ToTensor()(img_pil).cuda()[None, ...]
    # [batch, H, W]
    output = evaluator.evaluate(img, return_full=True)

    current_cmap = copy.copy(cm.get_cmap("jet"))
    w = img_pil.shape[1]
    h = img_pil.shape[0]
    cv2.imwrite(os.path.join(out_dir, "input_image.png"), img_pil[:,:,::-1])
    segm_map = (255*decode_seg_map_sequence(torch.max(output["segmentation"], 1)[1].detach().cpu().numpy(), dataset="cityscapes").numpy()).astype(np.uint8)[0,...]
    cv2.imwrite(os.path.join(out_dir, "semantic_segmentation.png"), segm_map.transpose(1, 2, 0)[:,:,::-1])
    
    anomaly = (current_cmap(output["anomaly_score"].cpu().numpy())[...,:3]*255).astype(np.uint8)[0,0,...,::-1]
    cv2.imwrite(os.path.join(out_dir, "anomaly_score.png"), anomaly)

    recon_img = cv2.resize((255*output["recon_img"][0, ...].detach().cpu().numpy()).transpose(1, 2, 0).astype(np.uint8), (w, h))[:,:,::-1]
    cv2.imwrite(os.path.join(out_dir, "recon_img.png"), recon_img)
    err_img = (255*current_cmap(output["recon_loss"][0, 0, ...].detach().cpu().numpy())).astype(np.uint8)[:, :, :-1]
    cv2.imwrite(os.path.join(out_dir, "recon_loss.png"), cv2.resize(err_img, (w, h))[:,:,::-1])

    inpainting = np.clip(255*output["inpainting"].cpu().numpy()[0, ...].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "inpainting_image.png"), inpainting[:,:,::-1])

    inpainting_mask = 255*(output["inpainting_mask"].cpu().numpy()[0, 0,...] > 0)
    print ("inpainting_mask: ", inpainting_mask.shape)
    cv2.imwrite(os.path.join(out_dir, "inpainting_mask.png"), inpainting_mask)

    perceptual_loss = 255*(current_cmap(output["perceptual_loss"].cpu().numpy()))[0, 0, ..., :3]
    print ("perceptual_loss: ", perceptual_loss.shape)
    cv2.imwrite(os.path.join(out_dir, "perceptual_loss.png"), perceptual_loss[:,:,::-1])


if __name__ == "__main__":
    main()
