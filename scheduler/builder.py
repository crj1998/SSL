""" 
This file build a scheduler for semi supervised learning from `scheduler` field of config.yaml 

scheduler:
  name: str
"""

import math
from copy import deepcopy
from functools import partial

# from .cosine_with_warmup import cosine_schedule_with_warmup


from torch.optim.lr_scheduler import LambdaLR

def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0., math.cos(math.pi * num_cycles * progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


SCHEDULER = {
    "cosine_schedule_with_warmup": cosine_schedule_with_warmup
}


def build(config):
    params = deepcopy(config)
    name = params.pop("name")
    return partial(SCHEDULER[name], **params)