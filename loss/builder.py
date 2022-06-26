"""
This file builds the criterion loss for the framework
"""
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F

LOSS = {
    "cross_entropy": F.cross_entropy,
    "mae": F.l1_loss,
    "l1": F.l1_loss,
    "mse": F.mse_loss,
    "l2": F.mse_loss,
}


def build(config):
    params = deepcopy(config)
    name = params.pop("name")
    return partial(LOSS[name], **params)
