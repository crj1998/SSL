""" 
FixMatch
The performance is similar to FixMatch. Thanks to https://github.com/kekmodel/FixMatch-pytorch.
"""

import math
import os

import torch
import torch.nn.functional as F
from loss import builder as loss_builder

from .base_trainer import Trainer


class FixMatch(Trainer):
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super(FixMatch, self).__init__(cfg, device)

        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        self.threshold = cfg.get("threshold", 0.95)

        self.threshold = 0.95
        self.T = cfg.get("T", 1.0)
        # pseudo with ema, this will intrige bad results and not used in paper
        self.pseudo_with_ema = all_cfg.get("ema", False) and all_cfg.ema.get("pseudo_with_ema", False)
        # distribution alignment mentioned in paper
        self.prob_list = []
        self.da = cfg.get("DA", False)
        self.da_len = cfg.get("DA_len", 256) if self.da else 0

    @torch.no_grad()
    def _da_pseudo_label(self, prob_list, logits_u_w):
        """ distribution alignment
        """
        probs = torch.softmax(logits_u_w.detach(), dim=1)

        prob_list.append(probs.mean(dim=0))
        if len(prob_list) > self.da_len:
            prob_list.pop(0)
        prob_avg = torch.stack(prob_list, dim=0).mean(dim=0)
        probs = probs / prob_avg
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def compute_loss(self, data_x, data_u, model, ema_model=None, **kwargs):
        # make inputs
        inputs_x, targets_x = data_x
        inputs_u_w, inputs_u_s, targets_u = data_u

        batch_size = inputs_x.size(0)
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        inputs_u_w, inputs_u_s, targets_u = inputs_u_w.to(self.device), inputs_u_s.to(self.device), targets_u.to(self.device)

        # whether use ema on pseudo labels
        if self.pseudo_with_ema:
            # for ema pseudo label
            logits_x = model(inputs_x)
            logits_u_s = model(inputs_u_s)
            logits_u_w = ema_model(inputs_u_w)
        else:
            inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
            logits = model(inputs)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(chunks=2, dim=0)
            del logits

        # inputs = torch.cat([inputs_x, inputs_u_w], dim=0)
        # logits = model(inputs, False)
        # logits_x, logits_u_w = logits[:batch_size], logits[batch_size:]
        # logits_u_s = model(inputs_u_s, True)
        
            
        # whether use da for pseudo labels, the performance is also lower based on our performance
        if self.da:
            probs_u_w = self._da_pseudo_label(self.prob_list, logits_u_w)
        else:
            probs_u_w = torch.softmax(logits_u_w.detach() / self.T, dim=-1)

        # making pseudo labels
        p_targets_u, mask = self._get_psuedo_label_and_mask(probs_u_w)
        
        # weighted = torch.gather(probs_u_w, dim=-1, index=p_targets_u.reshape(-1, 1))
        # beta = 4.0
        # semi-supervised loss
        Lu = (mask * self.loss_u(logits_u_s, p_targets_u, reduction='none')).mean()
        # pseudo = F.one_hot(p_targets_u, num_classes=10)
        # Lu = (mask * (F.l1_loss(F.softmax(logits_u_s, dim=-1), pseudo, reduction='none').sum(dim=-1) + self.loss_u(logits_u_s, p_targets_u, reduction='none')) ).mean()
        # Lu = (mask * (self.loss_x(logits_u_w, p_targets_u, reduction='none') + beta * F.kl_div(F.log_softmax(logits_u_s, dim=-1), F.softmax(logits_u_w.detach(), dim=-1), reduction="none").sum(dim=-1))).mean()
        # supervised loss
        Lx = self.loss_x(logits_x, targets_x, reduction='mean')

        loss = Lx + self.cfg.lambda_u * Lu

        # calculate pseudo label acc
        pseudo_label_acc = self._get_pseudo_label_acc(p_targets_u, mask, targets_u)

        loss_dict = {
            "loss": loss.item(),
            "loss_x": Lx.item(),
            "loss_u": Lu.item(),
            "mask_prob": mask.mean().item(),
            "pseudo_acc": pseudo_label_acc.item(),
        }
        return loss, loss_dict
