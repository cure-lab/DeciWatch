import argparse
from yacs.config import CfgNode as CN
import os

# CONSTANTS
# You may modify them at will
BASE_DATA_DIR = 'data'  # data dir
PW3D_DETECTED_PATH = os.path.join(
    BASE_DATA_DIR,
    'detected_poses/pw3d')  # the folder for 3DPW dataset detected poses
PW3D_GROUND_TRUTH_PATH = os.path.join(
    BASE_DATA_DIR,
    'groundtruth_poses/pw3d')  # the folder for 3DPW dataset groundtruth poses
H36M_DETECTED_PATH = os.path.join(
    BASE_DATA_DIR,
    'detected_poses/h36m')  # the folder for Human3.6M dataset detected poses
H36M_GROUND_TRUTH_PATH = os.path.join(
    BASE_DATA_DIR, 'groundtruth_poses/h36m'
)  # the folder for Human3.6M dataset groundtruth poses
JHMDB_DETECTED_PATH = os.path.join(
    BASE_DATA_DIR,
    'detected_poses/jhmdb')  # the folder for Sub-JHMDB dataset detected poses
JHMDB_GROUND_TRUTH_PATH = os.path.join(
    BASE_DATA_DIR, 'groundtruth_poses/jhmdb'
)  # the folder for Sub-JHMDB dataset groundtruth poses
AIST_DETECTED_PATH = os.path.join(
    BASE_DATA_DIR,
    'detected_poses/aist')  # the folder for AIST++ dataset detected poses
AIST_GROUND_TRUTH_PATH = os.path.join(
    BASE_DATA_DIR, 'groundtruth_poses/aist'
)  # the folder for AIST++ dataset groundtruth poses

# Configuration variables
cfg = CN()
cfg.DEVICE = 'cuda'  # training device 'cuda' | 'cpu'
cfg.SEED_VALUE = 4321  # random seed
cfg.LOGDIR = ''  # log dir
cfg.EXP_NAME = 'default'  # experiment name
cfg.DEBUG = True  # debug
cfg.OUTPUT_DIR = 'results'  # output folder

cfg.DATASET_NAME = ''  # dataset name
cfg.ESTIMATOR = ''  # backbone estimator name
cfg.BODY_REPRESENTATION = ''  # 3D | 2D | smpl
cfg.SAMPLE_INTERVAL = 10  # sample interval

cfg.SMPL_MODEL_DIR = "data/smpl/"  # smpl model dir

# CUDNN config
cfg.CUDNN = CN()  # cudnn config
cfg.CUDNN.BENCHMARK = True  # cudnn config
cfg.CUDNN.DETERMINISTIC = False  # cudnn config
cfg.CUDNN.ENABLED = True  # cudnn config

# dataset config
cfg.DATASET = CN()
cfg.DATASET.PW3D = CN()  # 3DPW dataset config
cfg.DATASET.PW3D.GROUND_TRUTH_PATH = PW3D_GROUND_TRUTH_PATH  # 3DPW dataset groundtruth path
cfg.DATASET.PW3D.DETECTED_PATH = PW3D_DETECTED_PATH  # 3DPW dataset detected pose path
cfg.DATASET.PW3D.KEYPOINT_NUM = 14  # keypoint number of dataset 3DPW, here we only evaluate on 14 joints following original setting
cfg.DATASET.PW3D.KEYPOINT_ROOT = [2, 3]  # 3DPW dataset keypoint root
cfg.DATASET.H36M = CN()  # Human3.6M dataset config
cfg.DATASET.H36M.GROUND_TRUTH_PATH = H36M_GROUND_TRUTH_PATH  # Human3.6M dataset groundtruth path
cfg.DATASET.H36M.DETECTED_PATH = H36M_DETECTED_PATH  # Human3.6M dataset detected pose path
cfg.DATASET.H36M.KEYPOINT_NUM = 17  # Human3.6M dataset keypoint number
cfg.DATASET.H36M.KEYPOINT_ROOT = [0]  # Human3.6M dataset keypoint root
cfg.DATASET.JHMDB = CN()  # Sub-JHMDB dataset config
cfg.DATASET.JHMDB.GROUND_TRUTH_PATH = JHMDB_GROUND_TRUTH_PATH  # Sub-JHMDB dataset groundtruth path
cfg.DATASET.JHMDB.DETECTED_PATH = JHMDB_DETECTED_PATH  # Sub-JHMDB dataset detected pose path
cfg.DATASET.JHMDB.KEYPOINT_NUM = 15  # Sub-JHMDB dataset keypoint number
cfg.DATASET.JHMDB.KEYPOINT_ROOT = [2]  # Sub-JHMDB dataset keypoint root
cfg.DATASET.AIST = CN()  # AIST++ dataset config
cfg.DATASET.AIST.GROUND_TRUTH_PATH = AIST_GROUND_TRUTH_PATH  # AIST++ dataset groundtruth path
cfg.DATASET.AIST.DETECTED_PATH = AIST_DETECTED_PATH  # AIST++ dataset detected pose path
cfg.DATASET.AIST.KEYPOINT_NUM = 14  # keypoint number of dataset AIST++, here we only evaluate on 14 joints following original setting
cfg.DATASET.AIST.KEYPOINT_ROOT = [2, 3]  # AIST++ dataset keypoint root

# model config
cfg.MODEL = CN()
cfg.MODEL.TYPE = 'network'  # 'network', 'linear', 'quadratic', 'spline'
cfg.MODEL.NAME = ''  # Used for saving the model
# sampling setting
cfg.MODEL.SAMPLE_TYPE = 'uniform'  # 'uniform', 'random', 'rand-uni'
cfg.MODEL.SLIDE_WINDOW_Q = 10  # Q = frames sampled in one slide window -1
cfg.MODEL.INTERVAL_N = cfg.SAMPLE_INTERVAL  # sampling interval N
cfg.MODEL.SLIDE_WINDOW_SIZE = cfg.MODEL.INTERVAL_N * cfg.MODEL.SLIDE_WINDOW_Q + 1  # slide window size
cfg.MODEL.SLIDE_WINDOW = True  # use slide window
cfg.MODEL.DROPOUT = 0.1  # dropout rate

# training config
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 1024  # batch size
cfg.TRAIN.WORKERS_NUM = 0  # workers number
cfg.TRAIN.EPOCH = 70  # epoch number
cfg.TRAIN.LR = 0.001  # learning rate
cfg.TRAIN.LRDECAY = 0.99  # learning rate decay rate
cfg.TRAIN.RESUME = None  # resume training checkpoint path
cfg.TRAIN.VALIDATE = True  # validate while training
cfg.TRAIN.USE_SMPL_LOSS = False  # True: use 3D keypoint as supervision | False: use pose parameter as supervivion
cfg.TRAIN.USE_6D_SMPL = True  # True: use 6D rotation | False: use Rotation Vectors (only take effect when cfg.TRAIN.USE_SMPL_LOSS=False )
cfg.TRAIN.PRE_NORM = False  # pre-norm in model

# test config
cfg.EVALUATE = CN()
cfg.EVALUATE.PRETRAINED = ''  # evaluation checkpoint
cfg.EVALUATE.ROOT_RELATIVE = True  # root relative represntation in error caculation
cfg.EVALUATE.SLIDE_WINDOW_STEP_Q = 10  # slide window step
cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE = cfg.MODEL.INTERVAL_N * cfg.EVALUATE.SLIDE_WINDOW_STEP_Q  # slide window step size
cfg.EVALUATE.INTERP='linear'
cfg.EVALUATE.RELATIVE_IMPROVEMENT=False
cfg.EVALUATE.DENOISE=False

# loss config
cfg.LOSS = CN()
cfg.LOSS.LAMADA = 2.0  # loss lamada
cfg.LOSS.W_DENOISE = 1.0  # loss w denoise

# encoder
cfg.MODEL.ENCODER_RESIDUAL = True  # encoder residual
cfg.MODEL.ENCODER_HEAD = 4  # encoder multi-head attention head number
cfg.MODEL.ENCODER_TRANSFORMER_BLOCK = 3  # encoder transformer block number
cfg.MODEL.ENCODER_EMBEDDING_DIMENSION = 512  # encoder embedding size

# decoder
cfg.MODEL.DECODER = 'transformer'  # 'transformer', 'tradition_interp'
cfg.MODEL.DECODER_INTERP = 'linear'  # 'linear', 'bilinear', 'bicubic'
cfg.MODEL.DECODER_RESIDUAL = True  # decoder residual
cfg.MODEL.DECODER_HEAD = 4  # decoder head number
cfg.MODEL.DECODER_TRANSFORMER_BLOCK = 3  # decoder transformer block number
cfg.MODEL.DECODER_EMBEDDING_DIMENSION = 512  # decoder embedding size
cfg.MODEL.DECODER_TOKEN_WINDOW = 5  # decoder token window

# visualization config
cfg.VIS = CN()
cfg.VIS.INPUT_VIDEO_NUMBER = 0 # visualization number
cfg.VIS.INPUT_VIDEO_PATH = 'data/videos/' # folder for input dataset images
cfg.VIS.OUTPUT_VIDEO_PATH = 'demo/' # output path
cfg.VIS.START = 0 # start frame
cfg.VIS.END = 1000 # end frame

# log config
cfg.LOG = CN()
cfg.LOG.NAME = ''  # log name


def get_cfg_defaults():
    """Get yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='dataset name [pw3d, h36m, jhmdb, pw3d]')
    parser.add_argument(
        '--estimator',
        type=str,
        help='backbone estimator name [spin, eft, pare, pw3d, fcn, simplepose]'
    )
    parser.add_argument('--body_representation',
                        type=str,
                        help='human body representation [2D, 3D, smpl]')
    parser.add_argument('--sample_interval',
                        type=int,
                        help='sampling ineterval N')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    cfg.DATASET_NAME = args.dataset_name
    cfg.ESTIMATOR = args.estimator
    cfg.BODY_REPRESENTATION = args.body_representation
    cfg.SAMPLE_INTERVAL = args.sample_interval
    cfg.MODEL.INTERVAL_N = cfg.SAMPLE_INTERVAL

    # cfg.MODEL.SLIDE_WINDOW_Q=10//cfg.MODEL.INTERVAL_N
    # cfg.EVALUATE.SLIDE_WINDOW_STEP_Q=cfg.MODEL.SLIDE_WINDOW_Q

    cfg.MODEL.SLIDE_WINDOW_SIZE = cfg.MODEL.INTERVAL_N * cfg.MODEL.SLIDE_WINDOW_Q + 1
    cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE = cfg.MODEL.INTERVAL_N * cfg.EVALUATE.SLIDE_WINDOW_STEP_Q

    return cfg, cfg_file