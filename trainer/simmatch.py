""" 
Self-implemented SimMatch
Thanks to https://github.com/KyleZheng1997/simmatch.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from loss import builder as loss_builder
from .base_trainer import Trainer



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SimMatch(Trainer):
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super(SimMatch, self).__init__(cfg, device)

        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        self.lambda_u = cfg.get("lambda_u", 1.0)
        self.lambda_in = cfg.get("lambda_in", 1.0)

        self.threshold = cfg.get("threshold", 0.95)
        self.DA = cfg.get("DA", False)
        self.c_smooth = cfg.get("smooth", 0.7)
        self.tt = cfg.get("tt", 1.0)
        self.st = cfg.get("st", 1.0)
        self.bank_m = cfg.get("bank_m", 0.7)

        self.num_classes = 10
        self.K = 256
        self.dim = cfg.get("low_dim", 128)

        # create the queue
        self.bank = F.normalize(torch.randn(self.dim, self.K, device=self.device), dim=0)
        self.labels = torch.zeros(self.K, dtype=torch.long, device=self.device)

        if self.DA:
            self.DA_len = cfg.get("DA_len", 256)
            self.DA_queue = torch.zeros(self.DA_len, self.num_classes, dtype=torch.float, device=self.device)
            self.DA_ptr = 0

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.detach().mean(dim=0)
        ptr = int(self.DA_ptr)
        # if torch.distributed.get_world_size() > 1:
        #     torch.distributed.all_reduce(probs_bt_mean)
        #     self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        # else:
        #     self.DA_queue[ptr] = probs_bt_mean
        self.DA_queue[ptr] = probs_bt_mean
        self.DA_ptr = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(dim=0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @torch.no_grad()
    def _update_bank(self, k, labels, index):
        # if torch.distributed.get_world_size() > 1:
        #     k      = concat_all_gather(k)
        #     labels = concat_all_gather(labels)
        #     index  = concat_all_gather(index)
        self.bank[:, index] = F.normalize(self.bank[:, index] * self.bank_m +  k.t() * (1-self.bank_m))
        self.labels[index] = labels

    def compute_loss(self, data_x, data_u, model, ema_model=None, **kwargs):
        # make inputs
        start_unlabel = kwargs.get("epoch") and kwargs.get("epoch")>0
        im_x, targets_x, index = data_x
        im_u_w, im_u_s, targets_u = data_u

        batch_x = im_x.size(0)
        batch_u = im_u_w.size(0)

        im_x, targets_x = im_x.to(self.device), targets_x.to(self.device)
        im_u_w, im_u_s, targets_u = im_u_w.to(self.device), im_u_s.to(self.device), targets_u.to(self.device)

        bank = self.bank.clone().detach()


        logits, embedding = model(torch.cat([im_x, im_u_w, im_u_s], dim=0))
        logits_x, logits_u_w, logits_u_s = logits[:batch_x], logits[batch_x: batch_x+batch_u], logits[batch_x+batch_u:]
        embedding_x, embedding_u_w, embedding_u_s = embedding[:batch_x], embedding[batch_x: batch_x+batch_u], embedding[batch_x+batch_u:]
        
        prob_u_w = F.softmax(logits_u_w, dim=-1)

        if self.DA:
            prob_u_w = self.distribution_alignment(prob_u_w)

        if start_unlabel:
            with torch.no_grad():
                teacher_logits = embedding_u_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.tt, dim=1)
                
                factor = prob_u_w.gather(1, self.labels.expand([batch_u, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.c_smooth < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels.expand([bs,-1]) , teacher_prob_orig)
                    prob_u_w = prob_u_w * self.c_smooth + aggregated_prob * (1-self.c_smooth)
            student_logits = embedding_u_s @ bank
            student_prob = F.softmax(student_logits / self.st, dim=1)
            loss_in = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
        else:
            loss_in = torch.tensor(0, dtype=torch.float, device=self.device)

        
        self._update_bank(embedding_x, targets_x, index)

        # making pseudo labels
        p_targets_u, mask = self._get_psuedo_label_and_mask(prob_u_w)
        
        loss_x = self.loss_x(logits_x, targets_x, reduction='mean')
        # loss_u = (mask * self.loss_u(logits_u_s, p_targets_u, reduction='none')).mean()
        loss_u = (torch.sum( - F.log_softmax(logits_u_s, dim=-1) * prob_u_w.detach(), dim=-1) * mask).mean()

        loss = loss_x + self.lambda_u * loss_u + self.lambda_in * loss_in

        pseudo_label_acc = self._get_pseudo_label_acc(p_targets_u, mask, targets_u)

        loss_dict = {
            "loss": loss,
            "loss_x": loss_x,
            "loss_u": loss_u,
            "loss_in": loss_in, 
            "mask_prob": mask.mean(),
            "pseudo_acc": pseudo_label_acc,
        }
        return loss, loss_dict