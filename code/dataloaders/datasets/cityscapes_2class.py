import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CityscapesSegmentation_2Class(data.Dataset):
    NUM_CLASSES = 2 

    def __init__(self, cfg, root, split):

        self.root = root
        self.split = split
        self.files = {}
        self.base_size = cfg.INPUT.BASE_SIZE
        self.crop_size = cfg.INPUT.CROP_SIZE
        self.img_mean = cfg.INPUT.NORM_MEAN 
        self.img_std = cfg.INPUT.NORM_STD
        self.road_roi_only = cfg.DATASET.ROAD_ROI_ONLY

        if cfg.DATASET.PERSPECTIVE_AUG:
            aug = tr.RandomRoadObjectScale(use=cfg.AUG.RANDOM_ANOMALY, rcp=cfg.AUG.RANDOM_CROP_PROB) 
        #elif self.road_roi_only:
        #    aug = tr.RandomRoadObjectScaleROI(use=cfg.AUG.RANDOM_ANOMALY, rcp=cfg.AUG.RANDOM_CROP_PROB) 
        else:
            aug = tr.RandomRoadObject(use=cfg.AUG.RANDOM_ANOMALY, rcp=cfg.AUG.RANDOM_CROP_PROB) 


        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 16, 29, 30, -1,
                            11, 23, 255] # + building + sky - 14,15,18 (guard rail, bridge, polegroup)
        self.valid_classes = [7, 8]

        self.ignore_index = cfg.LOSS.IGNORE_LABEL
        self.class_map = dict(zip([0, 7], range(self.NUM_CLASSES)))

        self.transform_tr = transforms.Compose([
            tr.RandomSizeAndCrop(self.crop_size, False, pre_size=None, scale_min=cfg.AUG.SCALE_MIN,
                                           scale_max=cfg.AUG.SCALE_MAX, ignore_index=self.ignore_index),
            tr.RandomHorizontalFlip(),
            aug,
            tr.ColorJitter(brightness=cfg.AUG.COLOR_AUG, contrast=cfg.AUG.COLOR_AUG, 
                           saturation=cfg.AUG.COLOR_AUG, hue=cfg.AUG.COLOR_AUG),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.img_mean, std=self.img_std),
            tr.ToTensor()])
        self.transform_val = transforms.Compose([
            aug,
            tr.Resize(self.crop_size),
            tr.Normalize(mean=self.img_mean, std=self.img_std),
            tr.ToTensor()])
        self.transform_ts = transforms.Compose([
            tr.Resize(self.crop_size),
            tr.Normalize(mean=self.img_mean, std=self.img_std),
            tr.ToTensor()])


        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)

        if self.road_roi_only:
            mask = (_tmp == 1)
            mask_cols = np.where(np.sum(mask, axis=0)>0)[0]
            mask_rows = np.where(np.sum(mask, axis=1)>0)[0]
            if mask_cols.size > 0 and mask_rows.size > 0:
                xmin, xmax = np.min(mask_cols), np.max(mask_cols)
                ymin, ymax = np.min(mask_rows), np.max(mask_rows)
                valid_mask = np.zeros_like(_tmp)
                valid_mask[ymin:ymax+1, xmin:xmax+1] = 1
                _tmp[np.logical_not(valid_mask>0)] = self.ignore_index

        _target = Image.fromarray(_tmp.astype(np.uint8))
        
        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample["randseed"] = index
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)

        sample["image_name"] = os.path.basename(img_path)[:-4]
        return sample

    def encode_segmap(self, mask):
        # Put all void classes to zero
        mask_tmp = np.zeros_like(mask)
        for _voidc in self.void_classes:
            mask_tmp[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask_tmp[mask == _validc] = 1 
        return mask_tmp

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]



def main():
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from yacs.config import CfgNode as CN
    import cv2
    
    cfg = CN()
    cfg.INPUT = CN()
    cfg.INPUT.BASE_SIZE = 896    
    cfg.INPUT.CROP_SIZE = 896 
    cfg.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std) 
    cfg.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std) 
    cfg.AUG = CN()
    cfg.AUG.RANDOM_CROP_PROB = 0.5
    cfg.AUG.SCALE_MIN = 0.5
    cfg.AUG.SCALE_MAX = 2.0
    cfg.AUG.COLOR_AUG = 0.25
    cfg.AUG.RANDOM_ANOMALY = True       # use random polygons as anomaly
    cfg.LOSS = CN()
    cfg.LOSS.IGNORE_LABEL = 255
    cfg.DATASET = CN()
    cfg.DATASET.ROAD_ROI_ONLY = False 
    cfg.DATASET.PERSPECTIVE_AUG = True

    cityscapes_train = CityscapesSegmentation_2Class(cfg, root='/mnt/datagrid/public_datasets/CityScapes/',  split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=20, shuffle=True, num_workers=1)
    np.random.seed(42)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = (decode_segmap(tmp, dataset='cityscapes')*255).astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            #plt.figure()
            #plt.title('display')
            #plt.subplot(211)
            #plt.imshow(img_tmp)
            #plt.subplot(212)
            #plt.imshow(segmap)
            cv2.imwrite(f"/home/vojirtom/Documents/test{jj:02d}.png", np.concatenate([img_tmp, segmap], axis=1)[:,:,::-1])

        if ii == 1:
            break

    #plt.show(block=True)

if __name__ == '__main__':
    main()
