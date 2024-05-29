# Image-Consistent Detection of Road Anomalies As Unpredictable Patches 
Pytorch implementation of our WACV 2023 paper with the pre-trained model used to generate the results presented in the publication.

**[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Vojir_Image-Consistent_Detection_of_Road_Anomalies_As_Unpredictable_Patches_WACV_2023_paper.pdf)
| [Supplementary
Video](https://drive.google.com/file/d/1uDXmdjTTItU1lNfKLF7uniLmznktRcQc/view?usp=share_link)** 

If you use this work please cite:
```latex
@InProceedings{Vojir_2023_WACV,
    author    = {Voj{\'\i}\v{r}, Tom\'a\v{s} and Matas, Ji\v{r}{\'\i}},
    title     = {Image-Consistent Detection of Road Anomalies As Unpredictable Patches},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5491-5500}
}
```

## Update 🔥🔥
- **2024.05.29 💥 New model available! It utilize DINOv2 backbone for feature extraction and semantic segmentation. See Model Section.** 

## Overview

The method consists of three main components:
1. A semantic segmentation network. Currently, a DeepLab v3 architecture is
   used (code adopted from
   [jfzhang95](https://github.com/jfzhang95/pytorch-deeplab-xception)
   distributed under MIT licence). The backbone is ResNet-101 pre-trained on
   ImageNet. It was trained on CityScapes dataset and the network weights are
   fixed.
2. An inpainting network adopted from
   [csqiangwen](https://github.com/csqiangwen/DeepFillv2_Pytorch) and merged to
   single file module.
3. An anomaly estimation network. It is a standalone module that uses the
   features extracted from the ResNet-101 backbone and the output of the
   segmentation network before softmax normalization.

The configuration of the network architecture, as published in WACV2023, is
defined in the configuration file `parameters.yaml` .  The specific model is
loaded dynamically based on its string name (see `MODEL.NET` variable). Set the
model to `DeepLabEmbeddingGlobalOnlySegmFullRes` for faster version without the
inpainting module - see paper for details. The network implementation is
located in `./net/models.py`. 

**DISCLAIMER: This is a research code. There is lot of unused and cluttered code. You running this code means you will not blame the 
author(s) if this breaks your stuff. This code is provided AS IS without warranty of any kind.**

## Training

All configurations of training is done through the configuration file
`./config/defauls.py` or saved configurations of particular network
configuration. To re-create the training of the proposed architecture use the
`parameters.yaml` as a configuration file. Change the training/val/testing data
sources if needed.

The training datasets are set in the `DATASET.TRAIN` and `DATASET.VAL`. They
are a string variables and currently can be from this list
`['cityscapes_2class', 'citybdd100k_2class', 'bdd100k_2class']` for training
and `['cityscapes_2class', 'citybdd100k_2class', 'bdd100k_2class', 'LaF']` for
validation.

### Running training script
The training can be run on specific GPU as (using default configuration from `./config/defauls.py`)
```sh
CUDA_VISIBLE_DEVICES=<GPU_ID> python3 train.py
```
or using custom settings, e.g. saved from custom experiment:
```sh
CUDA_VISIBLE_DEVICES=<GPU_ID> python3 train.py --exp_cfg="./path/to/config_file.yaml"
```

### Dataset

Currently only three labels are used, label 0 for anomaly, label 1 for road and
255 for void. The dataloaders needs to provide the gt segmentation using these
labels only. For example, see e.g.
`./dataloaders/datasets/cityscapes_2class.py`.

The datasets loaders are located in `./dataloaders/datasets/`. Each dataset has
its own dataloader class. 

The path to datasets data are stored in `./mypath.py` where the identification
is a string that is then used in the configuration file to set `DATASET.TRAIN` and
in the `./dataloaders/__init__.py` where the dataset are instantiated.

To add new dataset:
1. path to its data needs to be added to `./mypath.py`.  
2. dataloader class needs to be implemented and stored in `./dataloaders/datasets/`.
3. it instantiation needs to be defined in `./dataloaders/__init__.py` 
4. the `DATASET.TRAIN` or `DATASET.VAL` needs to be set to the new dataset name in the configuration file  

## Testing

For the testing, the `./ReconAnon.py` script is used (see the file for
a minimal example). The `exp_dir` parameter needs to be set to point to a root
directory where the `code` directory and `parameters.yaml` are located.  The
inserted path on line 6 in `./ReconAnon.py` need to be set to point to the
`code/config` directory.  The `evaluate` function expect a tensor with size [1, C, H,
W] (i.e. batch size of 1) where the image is normalized into [0,1] range. 

## Models

There are three pre-trained models:
1. The semantic segmentation model (fixed, does not need to be modified).
   The path to the checkpoint `checkpoint-segmentation.pth` needs to be set
   in the configuration file `MODEL.RECONSTRUCTION.SEGM_MODEL` variable.
   Download from
   [gdrive_segmentation_model](https://drive.google.com/file/d/1ahx2EaYGQQpK5uXSBRagt_okFnvSex_I/view?usp=share_link).
2. The inpainting model (fixed, does not need to be modified).
   The path to the checkpoint `deepfillv2_WGAN_G_epoch40_batchsize4.pth` needs to be set
   in the configuration file `MODEL.INPAINT_WEIGHTS_FILE` variable.
   Download from
   [gdrive_inpaint_model](https://drive.google.com/file/d/1zb49M2dhRK_7RMPYhQA5l--WbKXFQVuw/view?usp=share_link).
3. Put the model of the anomaly detection network (either trained or use pre-trained)
   into `<GITREPO/code/checkpoints/>checkpoint-best.pth` or set path in
   the `parameters.yaml` file `MODEL.RESUME_CHECKPOINT` to **absolute path** to the
   checkpoint file. The model used in the publication was trained using
   the parameters provided in the `parameters.yaml` configuration file. It used
   CityScapes+BDD100k datasets for training and LaF training data for
   validation. Download from
   [gdrive_dacup_model](https://drive.google.com/file/d/1z-Hxfd8rqX1fSljZowVkeLC9w6w5xv3i/view?usp=share_link) (or model without the inpainting part [gdrive_dacup_w/o_inpaint_model](https://drive.google.com/file/d/1q_ZrQ9DfKtL-GcRL7UCtP-wsT5ZXmdyO/view?usp=sharing)).


**New anomaly detection model with DINOv2 backbone** is available
   [here](https://drive.google.com/file/d/1wq96czmP38P-8y8qvwOpzqZiU8eA6oZw/view?usp=sharing)
   with semantic segmentation part
   [here](https://drive.google.com/file/d/153ah9SqvwEeb3NK-O2Z2-wAxq9edxORt/view?usp=sharing)
   and the corresponding configuration is provided in `parameters_dinov2.yaml`.
   For setup follow the instruction from 3). **NOTE** that `ReconAnom.py` file
   for evaluation expects parameters in file `parameters.yaml` so if you want
   to use the DINOv2 model either modify `ReconAnom.py` file (around line
   31) or rename the `parameters_dinov2.yaml` to `parameters.yaml`.

## Performance 

The performance is evaluated on the road region using two pixel-wise metrics:
Average Precision (AP) = Area under Precision-Recall Curve, and False Positive
Rate @ 95% True Positive Rate (FPR@95) = False Positive Rate at operating point
where the True Positive Rate is 95%. In the Table the results are shown as AP
/ FPR@95 for each dataset. Note the significant improvement on the "harder"
datasets (RO, RO21).

|                               | LaF          | LaF-train    | FS           | RA           | RO            | OT            |
| ----------------------------- | ------------ | ------------ | ------------ | ------------ | ------------  | ------------- |
| JSR-Net (ICCV 2021)           | 79.4 / 4.3   | 87.8 / 1.7   | 79.3 / 4.7   | 93.4 / 8.9   | 79.8 / 0.9    | 28.1 / 28.7   |
| DaCUP w/o inpaint (WACV 2023) | 85.1 / 2.1   | ---          | 88.8 / 1.7   | 94.3 /   6.8 | 90.3 /   0.17 | ---           |
| DaCUP (WACV 2023)             | 84.5 / 2.6   | ---          | 89.7 / 1.4   | 96.2 /   5.5 | 94.3 /  0.08  | 81.5 / 1.1    |
| DINOv2 DaCUP                  | ---          | ---          | 93.3 / 0.6   | 98.6 / 2.4   | 90.7 /  0.4   | 83.6 / 1.4    |



Datasets used for evaluation:
* [0] LaF - Lost and Found dataset Testing split
* [0] LaF-train - Lost and Found dataset Training split (this was used as a validation dataset during training)
* [1] RA - RoadAnomaly
* [2] RO - RoadObstacles
* [3] OT - Obstacle Track 
* [4] FS - FishyScapes dataset (subset of Lost and Found, for backward results comparability)


[0] P. Pinggera, S. Ramos, S. Gehrig, U. Franke, C. Rother, and R. Mester. Lost
and Found: detecting small road hazards for self-driving vehicles. In
International Conference on Intelligent Robots and Systems (IROS), 2016.  

[1] K. Lis, K. Nakka, P. Fua, and M. Salzmann. Detecting the Unexpected via Image
Resynthesis. In Int. Conf. Comput.  Vis., October 2019.

[2] Krzysztof Lis, Sina Honari, Pascal Fua, and Mathieu Salzmann. Detecting
Road Obstacles by Erasing Them, 2020.

[3] [SegmentMeIfYouCan](https://segmentmeifyoucan.com/) benchmark

[4] H. Blum, P. Sarlin, J. Nieto, R. Siegwart, and C. Cadena.  Fishyscapes:
A Benchmark for Safe Semantic Segmentation in Autonomous Driving. In 2019
IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages
2403–2412, 2019.

# Licence
Copyright (c) 2021 Toyota Motor Europe<br>
Patent Pending. All rights reserved.

This work is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc/4.0/)

