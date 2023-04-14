from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_CPU = 4     
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.GPU_IDS = [0]                   # which gpus to use for training - list of int, e.g. [0, 1]
_C.SYSTEM.RNG_SEED = 42

_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet"                # choices: ['resnet', 'xception', 'drn', 'mobilenet']
_C.MODEL.OUT_STRIDE = 16                    # deeplab output stride
_C.MODEL.SYNC_BN = None                     # whether to use sync bn (for multi-gpu), None == Auto detect
_C.MODEL.FREEZE_BN = False                 

_C.MODEL.NET = "DeepLabEmbeddingGlobalOnlySegmFullResInpaint_wide"      # disable local patch decoder


_C.MODEL.RECONSTRUCTION = CN()
_C.MODEL.RECONSTRUCTION.LATENT_DIM = 4      # number of channels of latent space
_C.MODEL.RECONSTRUCTION.SEGM_MODEL = "/mnt/datagrid/personal/vojirtom/sod/mcsegm/20210302_234038_642070/checkpoints/checkpoint-best.pth"  #resnet 66.1
_C.MODEL.RECONSTRUCTION.SEGM_MODEL_NCLASS = 19  # 19 for cityscapes
_C.MODEL.RECONSTRUCTION.SKIP_CONN = True 
_C.MODEL.RECONSTRUCTION.SKIP_CONN_DIM = 32 
_C.MODEL.RECONSTRUCTION.ERROR_FNC = "ssim"  # ["vgg", "msssim+l1", "ssim", "l2"] reconstruction error function
_C.MODEL.RECONSTRUCTION.BLUR_IMG = True
_C.MODEL.RECONSTRUCTION.INTERPOLATION = "bilinear"

_C.MODEL.LOCAL_RECONSTRUCTION = CN()
_C.MODEL.LOCAL_RECONSTRUCTION.LATENT_DIM_PER_SCALE = 128    # 5*LATENT_DIM_PER_SCALE is input to FC
_C.MODEL.LOCAL_RECONSTRUCTION.EMBEDDING_DIM = 16            # size of embedding for each local patch
_C.MODEL.LOCAL_RECONSTRUCTION.SKIP_CONN = True 

_C.MODEL.INPAINT_WEIGHTS_FILE = "/mnt/datagrid/personal/vojirtom/sod/deepfillv2_WGAN_G_epoch40_batchsize4.pth"

_C.LOSS = CN()
_C.LOSS.IGNORE_LABEL = 255
_C.LOSS.SIZE_AVG = True
_C.LOSS.BATCH_AVG = True 
_C.LOSS.EMBEDDING_MARGIN = 1      # margin for contrastive/triplet/spherical losses 
_C.LOSS.WARM_UP_DYNAMIC_WEIGHTS = True 
_C.LOSS.HARD_NEGATIVE_START_EPOCH = -1      # from which epoch a hard negative mining should be used for the triplet loss (negative number disables hard negative). Note that by default semi-hard negatives are used for training.
_C.LOSS.DIRECTION_TRIPLET_REGULARIZATION = 0.2    # 0.0 ... disabled, else = gamma, good choice [0.2, 0.4), directional regularization in triplet loss https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.pdf
_C.LOSS.EMB_SPACE_REG = 50.0            
_C.LOSS.EMB_SPACE_REG_WEIGHT = 0.0      # 0.0 ... disabled
_C.LOSS.XENT_MARGIN_REG = False

_C.LOSS.TYPE = "GlobalOnlyDecL2TripletMarginLossXentCoupeling"
  
_C.DATASET = CN()
_C.DATASET.TRAIN = "citybdd100k_2class"      # choices: ['cityscapes_2class', 'bdd100k_2class', 'citybdd100k_2class'],
_C.DATASET.VAL = "LaF"                      # choices: ['cityscapes_2class', 'bdd100k_2class'],
_C.DATASET.TEST = "LaF"                     # choices: ['LaF'],
_C.DATASET.FULL_DATA = False
_C.DATASET.ROAD_ROI_ONLY = True 
_C.DATASET.PERSPECTIVE_AUG = True 


_C.EXPERIMENT= CN()
_C.EXPERIMENT.NAME = None                   # None == Auto name from date and time 
_C.EXPERIMENT.OUT_DIR = "./training/experiments/"
_C.EXPERIMENT.EPOCHS = 100                  # number of training epochs
_C.EXPERIMENT.START_EPOCH = 0
_C.EXPERIMENT.RESUME_CHECKPOINT = None      # path to resume file (stored checkpoint)
_C.EXPERIMENT.FINE_TUNE = False 
_C.EXPERIMENT.EVAL_INTERVAL = 1             # eval every X epoch
_C.EXPERIMENT.EVAL_METRIC = "AnomalyEvaluator" # available evaluation metrics from utils.metrics.py file
_C.EXPERIMENT.EVAL_USE_METRIC = True        # true = use evaluation metric, else use validation loss

_C.INPUT = CN()
_C.INPUT.BASE_SIZE = 896 
_C.INPUT.CROP_SIZE = 896 
_C.INPUT.NORM_MEAN = [0.485, 0.456, 0.406]  # mean for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.NORM_STD = [0.229, 0.224, 0.225]   # std for the input image to the net (image -> (0, 1) -> mean/std) 
_C.INPUT.BATCH_SIZE_TRAIN = 2            # None = Auto set based on training dataset
_C.INPUT.BATCH_SIZE_TEST = 2             # None = Auto set based on training batch size

_C.AUG = CN()
_C.AUG.RANDOM_CROP_PROB = 0.5               # prob that random polygon (anomaly) will be cut from image vs. random noise
_C.AUG.SCALE_MIN = 0.5
_C.AUG.SCALE_MAX = 2.0
_C.AUG.COLOR_AUG = 0.25
_C.AUG.RANDOM_ANOMALY = True       # use random polygons as anomaly

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.LR_SCHEDULER = "poly"          # choices: ['poly', 'step', 'cos']
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.NESTEROV = False



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()

