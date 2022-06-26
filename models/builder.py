""" This file builds model for the framework """

from copy import deepcopy
import torchvision.models as models
# from models import resnet, wideresnet
from models import resnet
from models import wrn as wideresnet
# from models import dev as network

MODEL = {
    "wideresnet": wideresnet.build,
    # "resnext": resnext.build,
    "resnet50": resnet.resnet50,
    "resnet18": resnet.resnet18,
    "resnet50": models.resnet50,
    # "network": network.build,
    # "ProjNet": network.ProjNet
}


def build(config):
    # params
    params = deepcopy(config)
    name = params.pop("name")

    # init model
    model = MODEL[name](**params)        
    return model
