from utils.ddp_trainer import DDPTrainer
import argparse
import os
import torch
import torch.distributed as dist
import random
import numpy as np
import datetime
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='vgg19_trans', help='the name of the model')
    parser.add_argument('--data-dir', default=r'E:\Dataset\Counting\UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=5*1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=600,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    parser.add_argument("--local_rank",type=int,default=-1,help='DDP param')
    parser.add_argument("--dist_url",default="env://",type=str,help='url used to set up ddp')
    args = parser.parse_args()
    return args

def get_envs():
    local_rank = int(os.getenv('LOCAL_RANK',-1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE',1))
    return local_rank, rank, world_size

def select_device(device):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()
        nd = len(device.strip().split(','))
    cuda = device !='cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device    

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True   

if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    args.local_rank, args.rank, args.world_size = get_envs()
    if args.local_rank != -1:
    #if True:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda',args.local_rank)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else "gloo",init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size,timeout=datetime.timedelta(seconds=7200))        
        

    trainer = DDPTrainer(args)
    trainer.setup()
    trainer.train()

    if args.world_size > 1 and args.rank == 0:
        dist.destroy_process_group()
