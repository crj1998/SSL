

from loss import builder as loss_builder
from .base_trainer import Trainer

"""
Supervised learning
"""

class Supervised(Trainer):
    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device)
        self.device = device
        self.loss_x = loss_builder.build(cfg.loss_x)

    def compute_loss(self, data_x, model, **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        logits = model(inputs_x)
        loss = self.loss_x(logits, targets_x)

        loss.backward()

        loss_dict = {
            "loss": loss,
        }
        return loss_dict
