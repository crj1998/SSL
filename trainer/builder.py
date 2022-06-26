""" 
This file build a dataset for semi supervised learning from `trainer` field of config.yaml 

train:
  eval_step: 1024
  total_steps: 1048576
  trainer: 
    name: str


"""

from copy import deepcopy
from functools import partial

from .supervised import Supervised
from .comatch import CoMatch
from .fixmatch import FixMatch

from .simmatch import SimMatch
from .devmatch import DevMatch
from .tradematch import TradeMatch

from .fumatch import FuMatch

# meta archs for all trainers
TRAINER = {
    "FixMatch": FixMatch,
    "CoMatch": CoMatch,
    "Supervised": Supervised,
    "SimMatch": SimMatch,
    "DevMatch": DevMatch,
    "TradeMatch": TradeMatch,
    "FuMatch": FuMatch
}


def build(config):
    params = deepcopy(config)
    name = params.pop("name")
    return partial(TRAINER[name], cfg=params)
