""" 
This file build a optimizer for semi supervised learning from `optimizer` field of config.yaml 

optimizer:
  name: str
  [optional] k: v
"""
from copy import deepcopy
import torch.optim as optim

from .lars_optimizer import LARS

OPTIMIZER = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}


def build(config, model):
    params = deepcopy(config)
    name = params.pop("name")

    use_lars = False
    if "lars" in params.keys():
        use_lars = params.pop("lars")

    no_decay = params.pop("no_decay", ['bias', 'bn'])
    grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': params.weight_decay
        }, {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }
    ]
    optimizer = OPTIMIZER[name](grouped_parameters, **params)

    if use_lars:
        optimizer = LARS(optimizer)
    return optimizer
