"""
this is the main sript for SSL experiment
"""

import os, time
import argparse, logging, yaml

import random
from calendar import c

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid


from dataset import builder as dataset_builder
from models import builder as model_builder
from optimizer import builder as optimizer_builder
from scheduler import builder as scheduler_builder
from trainer import builder as trainer_builder

from utils.logger import get_logger, get_writter
from utils.config import get_config
from utils.misc import colorstr, setup_seed, accuracy, AverageMeter, AverageMeterManeger, save_ckpt_dict


from contextlib import contextmanager

@contextmanager
def torch_distributed_zero_first(rank):
    """
    Decorator to make all processes in distributed training
    wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        dist.barrier(device_ids=[rank])
    yield
    if rank == 0:
        dist.barrier(device_ids=[0])


def get_test_model(ema_model, model, use_ema):
    """
    use ema model or test model
    """
    if use_ema:
        test_model = ema_model.ema
        test_prefix = "ema"
    else:
        test_model = model
        test_prefix = ""
    return test_model, test_prefix

@torch.no_grad()
def test(args, test_loader, model, epoch):
    """ Test function for model and loader
        when the model is ema model, will test the ema model
        when the model is model, will test the regular model
    """
    top1 = AverageMeter()
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        prec1, = accuracy(outputs, targets, topk=(1, ))
        top1.update(prec1.item(), inputs.size(0))
    return top1.avg


def resume(args, model, optimizer, scheduler, task_specific_info, ema_model=None):
    """
    resume from checkpoint
    """
    global best_acc
    if not os.path.isfile(args.resume):
        args.resume = os.path.join(args.out, "checkpoint.pth.tar")

    # try resume if specified
    if not os.path.isfile(args.resume):
        logger.info("Failed to resume from {}".format(args.resume))
    else:
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        for key in checkpoint.keys():
            if key not in ['epoch', 'state_dict', 'ema_state_dict', 'acc', 'best_acc', 'optimizer', 'scheduler']:
                task_specific_info[key] = checkpoint[key]
                try:
                    task_specific_info[key] = task_specific_info[key].to(args.device)
                except:
                    pass

from copy import deepcopy 


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, ema_model, optimizer, scheduler, trainer, task_specific_info):
    """
    train loop
    """
    global best_acc
    global board
    test_accs = []
    labeled_epoch = 0
    unlabeled_epoch = 0

    if args.world_size > 1:
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)


    for epoch in range(args.start_epoch, args.epochs):
        # init logger
        meter_manager = AverageMeterManeger()
        meter_manager.register('data_time')
        meter_manager.register('batch_time')
        model.train()
        
        end = time.time()
        for batch_idx in range(args.eval_step):
            try:
                data_x = labeled_iter.next()
            except Exception:
                labeled_epoch += 1
                if args.world_size > 1:
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                data_x = labeled_iter.next()

            try:
                data_u = unlabeled_iter.next()
            except Exception:
                unlabeled_epoch += 1
                if args.world_size > 1:
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                data_u = unlabeled_iter.next()

            meter_manager.data_time.update(time.time() - end)
            end = time.time()

            # calculate loss
            loss, loss_dict = trainer.compute_loss(
                data_x = data_x, data_u = data_u,
                model = model, ema_model = ema_model,
                epoch = epoch, batch = batch_idx,
                task_specific_info = task_specific_info
            )
            
            # whether to use amp for accelerated training
            if not isinstance(args.amp, bool) and args.amp is not None:
                with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif SCALER is not None:
                SCALER.scale(loss).backward()
            else:
                loss.backward()

            # step
            if SCALER is not None:
                SCALER.step(optimizer)
            else:
                optimizer.step()
            scheduler.step()

            # Updates the scale for next iteration
            if SCALER is not None:
                SCALER.update()

            # update ema if needed
            if args.use_ema:
                ema_model.update(model)

            model.zero_grad()

            meter_manager.batch_time.update(time.time() - end)
            end = time.time()

            # update logger
            meter_manager.try_register_and_update(loss_dict)


            if args.local_rank in [-1, 0]:
                iters = epoch * args.eval_step + batch_idx
                lr = round(scheduler.get_last_lr()[0], 3)
                board.add_scalar("Train/LR", lr, iters)
                board.add_scalar("Train/Loss", loss_dict.get("loss", 0.0), iters)
                board.add_scalar("Train/Loss_x", loss_dict.get("loss_x", 0.0), iters)
                board.add_scalar("Train/Loss_u", loss_dict.get("loss_u", 0.0), iters)
                board.add_scalar("Train/mask", loss_dict.get("mask_prob", 0.0), iters)
                board.add_scalar("Train/pseudo", loss_dict.get("pseudo_acc", 0.0), iters)
                if epoch & (epoch - 1) == 0 and batch_idx == 0:
                    board.add_image("img/weak", make_grid(data_u[0][:16], normalize=True, value_range=(-1.0, 1.0), nrow=4), epoch, dataformats = "CHW")
                    board.add_image("img/strong", make_grid(data_u[1][:16], normalize=True, value_range=(-1.0, 1.0), nrow=4), epoch, dataformats = "CHW")

                if (batch_idx+1) % args.interval == 0:
                    meter_desc = meter_manager.get_desc(["data_time", "batch_time"])
                    logger.debug(f"Train Iter: {epoch+1}, {batch_idx + 1:4d}. {meter_desc}")

        meter_desc = meter_manager.get_desc()
        logger.info(f"Train Epoch: {epoch+1}. LR: {scheduler.get_last_lr()[0]:.4f} {meter_desc}")

        if args.local_rank in [-1, 0]:
            # add test info
            test_model, test_prefix = get_test_model(ema_model, model, args.use_ema)
            test_acc = test(args, test_loader, test_model, epoch)
            test_accs.append(test_acc)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            lr = round(scheduler.get_last_lr()[0], 3)
            
            logger.info(f'Last/Best top-1 acc: {test_acc:.2f}%/{best_acc:.2f}%. Avg top-1 acc of last 10 epoch: {np.mean(test_accs[-10:]):.2f}%.')
            # save ckpt
            save_ckpt_dict(args, model, ema_model, epoch, test_acc, optimizer, scheduler, task_specific_info, is_best, best_acc)
            
            args.writer.update([epoch, f"{lr:.4f}", f"{meter_manager.loss.avg:.3f}", f"{meter_manager.loss_x.avg:.3f}", f"{meter_manager.loss_u.avg:.3f}", f"{test_acc:.2f}"])
            
            board.add_scalar("Test/Err", 100.0 - test_acc, epoch)
            

def main(args):
    # global variables
    global best_acc
    global SCALER
    global logger
    global board
    

    best_acc = 0
    # set up logger
    level = logging.DEBUG if "dev" in args.out else logging.INFO
    logger = get_logger(
        args = args,
        name = "SSL",
        level = level if args.local_rank in [-1, 0] else logging.WARN,
        fmt = "%(asctime)s [%(levelname)s] %(message)s",
        rank = args.local_rank
    )
    logger.debug(f"Get logger named {colorstr('SSL')}!")
    logger.debug(f"distributed available? {dist.is_available()}")

    args.writer = get_writter(args)
    args.writer.register(["Epoch", "LR", "Loss", "Lossx", "Lossu", "Acc"])
    
    # load config and override
    cfg = get_config(args.cfg, args.override)
    logger.debug(f"Load config file from {colorstr(args.cfg)}!")

    # save config to out_dir
    if args.local_rank in [-1, 0]:
        outfile = os.path.join(args.out, os.path.basename(args.cfg))
        with open(outfile, "w") as f:
            yaml.dump(eval(str(cfg)), f, allow_unicode=True, sort_keys=False)
        logger.debug(f"Save config file to {colorstr(outfile)}!")
        del outfile

    #setup random seed
    if args.seed and isinstance(args.seed, int):
        setup_seed(args.seed)
        logger.info(f"Setup random seed {colorstr('green', args.seed)}!")
    else:
        logger.info(f"Can not Setup random seed with seed is {colorstr('green', args.seed)}!")

    # set amp scaler, usually no use
    if args.fp16:
        SCALER = torch.cuda.amp.GradScaler()
    else:
        SCALER = None
    
    if cfg.get("amp", False) and cfg.amp.use:
        args.amp = True
        args.opt_level = cfg.amp.opt_level
    else:
        args.amp = False
    # logger.debug(f"Setup amp scaler!")
    
    # init dist params
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
        logger.debug(f"Single GPU used!")
    else:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.n_gpu = torch.cuda.device_count()
        args.local_rank = dist.get_rank()
        # torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        assert dist.is_initialized(), f"Distributed initialization failed!"
        logger.debug(f"Multi GPU used! Distributed initialization!")

    # set device
    args.device = device
    logger.debug(f"Current device: {device}")

    with torch_distributed_zero_first(args.local_rank):
        # make dataset
        labeled_dataset, unlabeled_dataset, test_dataset = dataset_builder.build(cfg.data)

        logger.info(f"Dataset {colorstr(cfg.data.name)} loaded. {colorstr('green', len(labeled_dataset))} labeled and {colorstr('green', len(unlabeled_dataset))} unlabeled samples for train, {colorstr('green', len(test_dataset))} samples for test!")


    # make dataset loader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    # prepare labeled_trainloader
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler = train_sampler(labeled_dataset),
        batch_size = cfg.data.batch_size//args.world_size,
        num_workers = cfg.data.num_workers,
        drop_last = True,
        pin_memory = True
    )
    # prepare unlabeled_trainloader
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler = train_sampler(unlabeled_dataset),
        batch_size = max(int(cfg.data.batch_size * cfg.data.mu), 1)//args.world_size,
        num_workers = cfg.data.num_workers,
        drop_last = True,
        pin_memory = True
    )
    # prepare test_loader
    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = cfg.data.batch_size,
        num_workers = cfg.data.num_workers,
        pin_memory = True
    )

    logger.info(f"Dataloader Initialized. Batch size: {colorstr('green', cfg.data.batch_size)}, Num workers: {colorstr('green', cfg.data.num_workers)}.")
    
    with torch_distributed_zero_first(args.local_rank):
        # build model
        model = model_builder.build(cfg.model)
        logger.info(f"Model: {colorstr(cfg.model.name)}. Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # load from pre-trained, before DistributedDataParallel constructor
        # pretrained is str and it exists and is file.
        if isinstance(args.pretrained, str) and os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            logger.debug(f"Start load pretrained weights @: {args.pretrained}.")
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            for k, v in state_dict.items():
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    newk = k[len("module.encoder_q."):]
                    state_dict[newk] = v
                del state_dict[k]

            # for k in list(state_dict.keys()):
            #     # retain only encoder_q up to before the embedding layer
            #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #         # remove prefix
            #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            #     # delete renamed or unused k
            #     del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            logger.warning(f"Missing keys {msg.missing_keys} in state dict.")

            logger.debug(f"Pretrained weights @: {args.pretrained} loaded!")


    model.to(args.device)

    # make optimizer,scheduler
    optimizer = optimizer_builder.build(cfg.optimizer, model)
    scheduler = scheduler_builder.build(cfg.scheduler)(optimizer)

    logger.info(f"Optimizer {colorstr(cfg.optimizer.name)} and Scheduler {colorstr(cfg.scheduler.name)} selected!")

    # set ema
    args.use_ema = False
    ema_model = None
    if cfg.get("ema", False) and cfg.ema.use:
        args.use_ema = True
        from models.ema import ModelEMA
        ema_model = ModelEMA(args.device, model, cfg.ema.decay)
        logger.info(f"EMA model with decay {colorstr('green', cfg.ema.decay)} used!")

    args.start_epoch = 0

    # initialize from resume for fixed info and task_specific_info
    task_specific_info = dict()

    if args.resume:
        resume(args, model, optimizer, scheduler, task_specific_info, ema_model)
    
    # builde model trainer
    cfg.train.trainer['amp'] = args.amp
    trainer = trainer_builder.build(cfg.train.trainer)(device=device, all_cfg=cfg)

    logger.info(f"Trainer {colorstr(cfg.train.trainer.name)} selected!")

    # process model
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        args.amp = amp
    
    # SyncBatchNorm if use BN in DDP
    if args.use_BN:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.local_rank != -1:
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters = True)

    args.total_steps = cfg.train.total_steps
    args.eval_step = cfg.train.eval_step
    args.epochs = args.total_steps // args.eval_step

    model.zero_grad()

    board = SummaryWriter(args.out)

    #train loop
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, ema_model, optimizer, scheduler, trainer, task_specific_info)

    board.close()



if __name__ == '__main__':
    """
    nohup python3.8 main.py --cfg configs/CIFAR100/tradematch.yaml --seed 42 --gpu 4 --out results/CIFAR100/TradeMatch2 --override train.trainer.beta=2.0 >c100tramatch2.log 2>&1 &
    nohup python3.8 main.py --cfg configs/CIFAR10/tradematch.yaml --seed 42 --gpu 7 --out results/CIFAR10/TradeMatch1 --override train.trainer.beta=1.0 >tramatch1.log 2>&1 &
    nohup python3.8 main.py --cfg configs/CIFAR10/tradematch.yaml --seed 42 --gpu 6 --out results/CIFAR10/TradeMatch2 --override train.trainer.beta=2.0 >tramatch2.log 2>&1 &
    nohup python3.8 main.py --cfg configs/CIFAR10/tradematch.yaml --seed 42 --gpu 5 --out results/CIFAR10/TradeMatch4 --override train.trainer.beta=4.0 >tramatch4.log 2>&1 &
    
    nohup python3.8 main.py --cfg configs/CIFAR10/tradematch.yaml --seed 42 --gpu 5 --out results/CIFAR10/TradeMatchsoft4 --override train.trainer.beta=4.0 >tramatch4.log 2>&1 &
    nohup python3.8 main.py --cfg configs/CIFAR10/tradematch.yaml --seed 42 --gpu 4 --out results/CIFAR10/TradeMatchsoft2 --override train.trainer.beta=2.0 >tramatch2.log 2>&1 &

    nohup python3.8 main.py --cfg configs/CIFAR10/fixmatch.yaml --seed 42 --gpu 4 --out results/CIFAR10/FixMatch >fixmatch.log 2>&1 &
    nohup python3.8 main.py --cfg configs/CIFAR10/fixmatch.yaml --seed 42 --gpu 7 --out results/CIFAR10FixMatchMAE+CE >maecefixmatch.log 2>&1 &
    python3.8 main.py --cfg configs/CIFAR100/fixmatch.yaml --seed 42 --gpu 4 --out results/CIFAR100FixMatch
    $ python main.py --cfg configs/fixmatch_cifar_wrn.yaml --seed 42 --gpu 0 --other-args "data.labeled.transforms.0.name=filp"
    python3.8 main.py --cfg configs/supervised.yaml --seed 5 --gpu 4
    nohup python3.8 main.py --seed 42 --gpu 2 --cfg configs/comatch_cifar_wrn.yaml --out results/CoMatch >comatch.log 2>&1 &
    nohup python3.8 main.py --seed 42 --gpu 0 --cfg configs/fixmatch_cifar_wrn.yaml --out results/FixMatch >fixmatch.log 2>&1 &
    $ python3.8 main.py --cfg configs/fixmatch_cifar_wrn.yaml --seed 42 --gpu 1 --out results/FixMatch2
    python3.8 main.py --cfg configs/fixmatch_cifar_wrn.yaml --seed 42 --gpu 5 --out results/FixMatchsum
    python3.8 main.py --cfg configs/comatch_cifar_wrn.yaml --seed 42 --gpu 3 --out results/dev2
    python3.8 main.py --cfg configs/dev.yaml --seed 42 --gpu 1 --out results/dev  >dev.log 2>&1 &
    nohup python3.8 main.py --cfg configs/fixmatch_ff.yaml --seed 42 --gpu 4 --out results/fixmatchFF >fixmatchFF.log 2>&1 &
    nohup python3.8 main.py --cfg configs/fixmatch_ff.yaml --seed 42 --gpu 7 --out results/fixmatchFFCut >fixmatchFFcut.log 2>&1 &
    nohup python3.8 main.py --cfg configs/simmatch_cifar_wrn.yaml --seed 42 --gpu 0 --out results/simmatch >simmatch.log 2>&1 &
    nohup python3.8 main.py --cfg configs/fixmatch_cifar_proj.yaml --seed 42 --gpu 7 --out results/FixMatchproj  >projmatch.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6,7 nohup python3.8 -m torch.distributed.launch --nproc_per_node=2 --master_port=23456 main.py --cfg configs/fixmatch_cifar_wrn.yaml --seed 42 --out results/fixmatchcutout >fixmatch.log 2>&1 &
    """
    
    
    parser = argparse.ArgumentParser(description='PyTorch Semi Supervised Learning')

    parser.add_argument('--cfg', type=str, required=True, help='a config')
    parser.add_argument('--gpu', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--out', default='results/supervised', help='directory to output the result')
    parser.add_argument('--interval', default=64, type=int, help='log interval')
    parser.add_argument('--pretrained', default=None, help='directory to pretrained model')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--use_BN', default=False, type=bool, help="use_batchnorm")
    parser.add_argument('--fp16', action='store_true', help="whether use fp16 for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        '--override', default='', type=str,
        help='overwrite the config, keys are split by space and args split by |, such as train.eval_step=2048|optimizer.lr=0.1'
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    main(args)


