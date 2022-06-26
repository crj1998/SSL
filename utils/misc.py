import os, shutil
import random
import numpy as np
import torch


def colorstr(*inputs):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
    string = str(string)
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']





def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def save_ckpt_dict(args, model, ema_model, epoch, test_acc, optimizer, scheduler, task_specific_info, is_best, best_acc):
    model_to_save = model.module if hasattr(model, "module") else model
    if args.use_ema:
        ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
    save_ckpt_dict = {
        'epoch': epoch + 1,
        'acc': test_acc,
        'best_acc': best_acc,
        'state_dict': model_to_save.state_dict(),
        'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    save_ckpt_dict.update(task_specific_info)
    save_checkpoint(save_ckpt_dict, is_best, args.out)


def setup_seed(seed):
    """ set seed for the whole program for removing randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
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

class HistgramMeter(object):
    """ Compute histgram meter
    """
    def __init__(self, default_len=10000):
        self.default_len = default_len
        self.reset()

    def reset(self):
        self._data_list = []

    @property
    def data_list(self):
        return np.array(self._data_list)

    def update(self, value):
        if len(self._data_list) > self.default_len:
            self._data_list = self._data_list[-self.default_len:]
        self._data_list.extend(list(value))
        
class AverageMeterManeger(object):
    def __init__(self):
        self.name_list = []

    def register(self, name, value=0):
        self.name_list.append(name)
        if isinstance(value, np.ndarray):
            setattr(self, name, HistgramMeter())
        else:
            setattr(self, name, AverageMeter())

    def reset_avgmeter(self):
        for idx, key in enumerate(self.name_list):
            if isinstance(getattr(self, key), AverageMeter):
                getattr(self, key).reset()

    def try_register_and_update(self, reg_data):
        for key, value in reg_data.items():
            if not hasattr(self, key):
                self.register(key, value)

            if isinstance(value, torch.Tensor):
                getattr(self, key).update(value.item())
            else:
                getattr(self, key).update(value)

    def get_desc(self, exclude=[]):
        meter_desc = ""
        for key in self.name_list:
            if isinstance( getattr(self, key), AverageMeter) and key not in exclude:
                meter_desc += "{}: {:.3f} ".format(key, getattr(self, key).avg)
        return meter_desc

    def add_to_writer(self, writer, epoch, prefix="train/"):
        for idx, key in enumerate(self.name_list):
            if isinstance(getattr(self, key), AverageMeter):
                writer.add_scalar("{}.{} ".format(prefix, key),
                                  getattr(self, key).avg, epoch)
            elif isinstance(getattr(self, key), HistgramMeter):
                writer.add_histogram("{}.{}".format(prefix, key),
                                     getattr(self, key).data_list, epoch)
            else:
                raise ValueError("Unsupported value type {}".format(
                    type(getattr(self, key))))