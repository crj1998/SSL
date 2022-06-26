import argparse
import builtins
import os, math, json, csv
import time

import numpy as np

from PIL import ImageFile        
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as T
import torchvision.models as models

from tqdm import tqdm

cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def torch_dist_sum(gpu, *args):
    process_group = torch.distributed.group.WORLD
    tensor_args = []
    pending_res = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg.clone().reshape(-1).detach().cuda(gpu)
        else:
            tensor_arg = torch.tensor(arg).reshape(-1).cuda(gpu)
        torch.distributed.all_reduce(tensor_arg, group=process_group)
        tensor_args.append(tensor_arg)
    
    return tensor_args

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def DDP_init(args):
    args.rank = -1
    args.local_rank = -1
    args.world_size = -1
    assert torch.cuda.is_available(), "DDP training requires GPU device."
    assert dist.is_available(), "DDP training is not available."

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f"Rank {local_rank} of {world_size} is initialized @ {master_addr}:{master_port}...")

    torch.cuda.set_device(local_rank)
    # communication backend, NCCL is recommended for NVIDIA GPUs.
    assert torch.distributed.is_nccl_available(), "backend NCCL is not available!"
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier()
    assert dist.is_available() and dist.is_initialized(), "DDP initialization failed!"
    args.rank        = rank
    args.local_rank  = local_rank
    args.world_size  = world_size
    args.distributed = True


# @torch.no_grad()
# def validate(dataloader, model, args):
#     top1 = AverageMeter('Acc@1', ':6.2f')

#     # switch to evaluate mode
#     model.eval()

#     for images, target in tqdm(dataloader, total=len(dataloader)):
#         # images = images.cuda(args.local_rank, non_blocking=True)
#         # target = target.cuda(args.local_rank, non_blocking=True)
#         images = images.cuda()
#         target = target.cuda()
#         # compute output
#         output = model(images)

#         # measure accuracy and record loss
#         acc1, *_ = accuracy(output, target, topk=(1, ))
#         top1.update(acc1[0], images.size(0))
    
#     if args.local_rank != -1:
#         sum1, cnt1 = torch_dist_sum(args.local_rank, top1.sum, top1.count)
#         top1_acc = sum1.float().sum() / cnt1.float().sum()
#         return top1_acc.item()
#     else:
#         return top1.avg


@torch.no_grad()
def validate(dataloader, model, num_classes=3):
    Probs = []
    for images, targets in tqdm(dataloader, total=len(dataloader)):
        images, targets = images.cuda(), targets.cuda()

        # compute output
        z = model(images)[:, :num_classes]
        p = F.softmax(z, dim=-1)
        Probs.append(p.cpu().numpy())

    # measure accuracy
    Probs = np.vstack(Probs)
    threshold = np.linspace(0.05, 0.95, 19).tolist()
    threshold.extend([0.975, 0.99, 0.995, 0.999])
    hits = []

    total = Probs.shape[0]
    print(f"Total: {total}")
    print("Thrs  Hit(rate)")
    print("----  ---------")
    for thrs in threshold:
        hit = np.any(Probs[:, 1:] > thrs, axis=-1).sum()
        print(f"{thrs:5.3f}  {hit:5d}({hit/total: .2%})")
        hits.append(hit/total)
    return threshold, hits

def main(args):
    if args.local_rank == -1:
        torch.cuda.set_device("cuda:0")
        args.world_size = 1
    else:
        DDP_init(args)

    # disable print except main processing
    if args.local_rank not in [-1, 0]:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    # prepare Dataset and Dataloader
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # "/home/hadoop-seccv/cephfs/data/zhaokang/projects/ssl/data/secure_white_pic_bsj/"
    dataset = ImageFolder(root=args.datafolder, transform=transform)    
    sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size = args.batch_size, shuffle=False, persistent_workers=True, num_workers = args.workers, pin_memory=False)
    
    
    print(f">>> Val: {len(dataset.samples)}")
    print(f">>> creating model '{args.model}'")
    # prepare model
    model = getattr(models, args.model)(num_classes=args.num_classes)
    if args.weights and os.path.exists(args.weights) and os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["ema_state_dict"])
        print(f">>> loaded weights @ {args.weights}")
    model.eval()

    if args.local_rank != -1:
        model.cuda(args.local_rank)
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters = True)
    else:
        model.cuda()



    acc1 = validate(dataloader, model, args.num_classes)
    # print(f'* Acc@1 {acc1:.3f}')

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # model parameters
    parser.add_argument('--model', metavar='MODEL', default='resnet50', help='model architecture')
    parser.add_argument('--num-classes', default=3, type=int, metavar='N', help='number of classes')
    parser.add_argument('--weights', default=".", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # dataset parameters
    parser.add_argument('--datafolder', default="./", type=str, metavar='PATH', help='path to dataset (default: pwd)')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size, this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')

    # distributed parameters
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument('--threshold', default=0.7, type=float, help='pseudo label threshold')

    args = parser.parse_args()
        
    main(args)

# CUDA_VISIBLE_DEVICES=5 python3.8 evaluate.py --model resnet50 --num-classes 3 --weights results/RedTheme/model_best.pth --datafolder /home/hadoop-seccv/ssd/rjchen/data/secure_white_pic_bsj/images_part0
# /home/hadoop-seccv/ssd/rjchen/data/redtheme_ssl_20220423to0427
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --lambda_in 5 --lr 0.03  --epochs 400 --warmup-epoch 5  --batch-size 32  --nesterov --cos --DA


# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --use_env evaluate.py


# 疑似 75371 98.10%