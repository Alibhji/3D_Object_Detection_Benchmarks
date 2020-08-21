
import os
from pathlib import Path

from tensorboardX import SummaryWriter


from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

import datetime

from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import torch
import glob
# from .test import repeat_eval_ckpt

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = {'cfg_file': '../cfgs/kitti_models/my_pointrcnn.yaml', 'batch_size': 2, 'epochs': None, 'workers': 1, 'extra_tag': 'default', 'ckpt': None, 'pretrained_model': None, 'launcher': 'none', 'tcp_port': 18888, 'sync_bn': False, 'fix_random_seed': False, 'ckpt_save_interval': 1, 'local_rank': 0, 'max_ckpt_save_num': 30, 'merge_all_iters_to_one_epoch': False, 'set_cfgs': None, 'max_waiting_mins': 0, 'start_epoch': 0, 'save_to_file': False}

args = Struct(**args)
print(args.cfg_file)


args.cfg_file= "./my_pointrcnn.yaml"
print(args.cfg_file)
cfg_from_yaml_file(args.cfg_file, cfg)
cfg.TAG = Path(args.cfg_file).stem
cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs, cfg)


if args.launcher == 'none':
    dist_train = False
    total_gpus = 1
else:
    total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
        args.tcp_port, args.local_rank, backend='nccl'
    )
    dist_train = True

if args.batch_size is None:
    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
else:
    assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    args.batch_size = args.batch_size // total_gpus

args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

if args.fix_random_seed:
    common_utils.set_random_seed(666)

output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
ckpt_dir = output_dir / 'ckpt'
output_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

# log to file
logger.info('**********************Start logging**********************')
gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

if dist_train:
    logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
for key, val in vars(args).items():
    logger.info('{:16} {}'.format(key, val))
log_config_to_file(cfg, logger=logger)
if cfg.LOCAL_RANK == 0:
    os.system('cp %s %s' % (args.cfg_file, output_dir))

tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

from pcdet.datasets import build_dataloader

train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from lib.dataset_tools import draw_point_cloud
import os
import cv2

image_path = train_set.root_split_path
frame=train_set[12]


image2 = os.path.join(image_path ,'image_2',frame['frame_id']+'.png' )
image2 = cv2.imread(image2)

image3 = os.path.join(image_path ,'image_3',frame['frame_id']+'.png' )
image3 = cv2.imread(image3)

f, ax = plt.subplots( 2, figsize=(15, 5))
ax[0].set_title('Left RGB Image (cam2)')
ax[0].imshow(image2)
ax[1].set_title('Right RGB Image (cam3)')
ax[1].imshow(image3)
# plt.show()

f2 = plt.figure(figsize=(15, 8))
ax2 = f2.add_subplot(111, projection='3d')
draw_point_cloud(frame,ax2, 'Velodyne scan', xlim3d=(-10,30))
# plt.show()





model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
#%%

if args.sync_bn:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.cuda()

optimizer = build_optimizer(model, cfg.OPTIMIZATION)

# load checkpoint if it is possible
start_epoch = it = 0
last_epoch = -1
if args.pretrained_model is not None:
    model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

if args.ckpt is not None:
    it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
    last_epoch = start_epoch + 1
else:
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        it, start_epoch = model.load_params_with_optimizer(
            ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1

model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
if dist_train:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
logger.info(model)

lr_scheduler, lr_warmup_scheduler = build_scheduler(
    optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
    last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
)

logger.info('**********************Start training %s/%s(%s)**********************'
            % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
train_model(
    model,
    optimizer,
    train_loader,
    model_func=model_fn_decorator(),
    lr_scheduler=lr_scheduler,
    optim_cfg=cfg.OPTIMIZATION,
    start_epoch=start_epoch,
    total_epochs=args.epochs,
    start_iter=it,
    rank=cfg.LOCAL_RANK,
    tb_log=tb_log,
    ckpt_save_dir=ckpt_dir,
    train_sampler=train_sampler,
    lr_warmup_scheduler=lr_warmup_scheduler,
    ckpt_save_interval=args.ckpt_save_interval,
    max_ckpt_save_num=args.max_ckpt_save_num,
    merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
)