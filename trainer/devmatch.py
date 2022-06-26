""" 
DevMatch for experiment
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import builder as loss_builder

from .base_trainer import Trainer

class CosineSimLoss(nn.Module):
    def __init__(self, dim=-1, reduction="mean", eps=1e-8):
        super(CosineSimLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, x, y):
        out = (1 - F.cosine_similarity(x, y, self.dim, self.eps))/2
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            return out


class CELoss(nn.Module):
    def __init__(self, dim=-1, reduction="mean", eps=1e-8):
        super(CELoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, x, y):
        out = - (F.log_softmax(x, dim=self.dim) * F.softmax(y.detach(), dim=self.dim)).sum(dim=self.dim)
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            return out

class NBCE(nn.Module):
    def __init__(self, dim=-1, reduction="mean", K=6):
        super(NBCE, self).__init__()
        self.dim = dim
        self.K = K
        self.reduction = reduction
        self.eps = 1e-5
    
    def forward(self, x):
        # k = self.K if isinstance(self.K, int) else int(x.size(-1)*self.K)
        k = int(self.K)
        indices = torch.topk(x.detach(), dim=self.dim, k=k, sorted=False, largest=False).indices
        targets = torch.zeros_like(x.detach())
        targets = targets.scatter(dim=self.dim, index=indices, value=1.0)
        out = (- torch.log(self.eps + 1.0 - F.softmax(x, dim=-1)) * targets).sum(dim=self.dim) / k
        if out.mean().isnan():
            print(out, F.softmax(x, dim=-1), targets)
            exit()
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            return out
        
        
        
class DevMatch(Trainer):
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super().__init__(cfg, device)
        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)
        # self.loss_u = CosineSimLoss(-1, "none")
        # self.loss_u = CELoss(-1, "none")
        self.loss_p = NBCE(-1, "none", 6)

        # pseudo with ema, this will intrige bad results and not used in paper
        self.pseudo_with_ema = False
        if all_cfg.get("ema", False):
            self.pseudo_with_ema = all_cfg.ema.get("pseudo_with_ema", False)

        self.threshold = cfg.get("threshold", 0.95)
        self.T = cfg.get("T", 1.0)
        self.lambda_u = cfg.get("lambda_u", 1.0)
        # distribution alignment mentioned in paper
        self.prob_list = []
        self.DA = cfg.get("DA", False)
        self.DA_len = cfg.get("DA_len", 256) if self.DA else 0
            

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
        batch_size_u = inputs_u_w.size(0)
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        inputs_u_w, inputs_u_s, targets_u = inputs_u_w.to(self.device), inputs_u_s.to(self.device), targets_u.to(self.device)


        inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
        logits, _, feats = model(inputs)
        logits_x, logits_u_w, logits_u_s = torch.split(logits, [batch_size, batch_size_u, batch_size_u], dim=0)
        feats_x, feats_u_w, feats_u_s = torch.split(feats, [batch_size, batch_size_u, batch_size_u], dim=0)


        # whether use da for pseudo labels, the performance is also lower based on our performance
        if self.DA:
            probs_u_w = self._da_pseudo_label(self.prob_list, logits_u_w.detach())
        else:
            probs_u_w = torch.softmax(logits_u_w.detach() / self.T, dim=-1)

        # making pseudo labels
        p_targets_u, mask = self._get_psuedo_label_and_mask(probs_u_w)
        if kwargs.get("epoch", 0) > 64:
            self.loss_p.K = 7
        elif kwargs.get("epoch", 0) > 100:
            self.loss_p.K = 8
            
        # semi-supervised loss
        Lu = (mask * self.loss_u(logits_u_s, p_targets_u, reduction='none') + (1.0 - mask) * self.loss_p(logits_u_s)).mean()
        # Lu = (mask * self.loss_x(logits_u_w, p_targets_u, reduction='none') + self.lambda_u * self.loss_u(feats_u_s, feats_u_w)).mean()
        # Lu = (mask * self.loss_x(logits_u_w, p_targets_u, reduction='none') + 2.0 * (feats_u_s - feats_u_w.detach()).abs().mean(dim=-1)).mean()
        # supervised loss
        Lx = self.loss_x(logits_x, targets_x, reduction='mean')

        loss = Lx +  Lu

        # calculate pseudo label acc
        pseudo_label_acc = self._get_pseudo_label_acc(p_targets_u, mask, targets_u)

        loss_dict = {
            "loss": loss,
            "loss_x": Lx,
            "loss_u": Lu,
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }
        return loss, loss_dict

"""
CUDA_VISIBLE_DEVICES=2,3 nohup python3.8 -m torch.distributed.launch --nproc_per_node=2 --master_port 66660 main.py --cfg configs/dev.yaml --seed 42 --out results/Devmatch >devmatch.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 python3.8 -m torch.distributed.launch --nproc_per_node=2 --master_port 66660 main.py --cfg configs/dev.yaml --seed 42 --out results/Devmatch

"""