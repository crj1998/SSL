
import torch
import torch.nn.functional as F
from loss import builder as loss_builder

import numpy as np
from .base_trainer import Trainer


class FuMatch(Trainer):
    def __init__(self, cfg, device, all_cfg, **kwargs):
        super(FuMatch, self).__init__(cfg, device)

        self.loss_x = loss_builder.build(cfg.loss_x)
        self.loss_u = loss_builder.build(cfg.loss_u)

        self.threshold = cfg.get("threshold", 0.95)

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

        with torch.no_grad():
            logits_u_w = model(inputs_u_w)
            probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, p_targets_u = probs_u_w.max(dim=-1)
            mask = max_probs > self.threshold

        vmin, vmax = 0.2, 0.8
        N, _, H, W = inputs_x.shape
        masks = torch.zeros_like(inputs_x)
        for j in range(N):
            if np.random.random() < 0.5:
                w = round(W*np.random.uniform(vmin, vmax))
                masks[j, :, :w, :] = 1.0
            else:
                h = round(H*np.random.uniform(vmin, vmax))
                masks[j, :, :, :h] = 1.0

        shuffle = torch.arange(batch_size, dtype=targets_x.dtype, device=targets_x.device)
        copy = False
        for c in targets_x.unique():
            mask_x = targets_x == c
            num_x = mask_x.sum().item()
            idx_w = (mask & (p_targets_u == c)).nonzero().squeeze(dim=-1)
            if num_x == 0 or idx_w.size(0) == 0:
                copy = True
                continue
            shuffle[mask_x] = idx_w[torch.randint(0, idx_w.size(0), (num_x, ))]

        if copy:
            inputs_inter = inputs_x.clone()
        else:
            inputs_inter = (1 - masks) * inputs_x + masks * inputs_u_w[shuffle]

        inputs = torch.cat([inputs_x, inputs_inter, inputs_u_w, inputs_u_s], dim=0)
        logits = model(inputs)
        logits_x, logits_inter = logits[:2*batch_size].chunk(chunks=2, dim=0)
        logits_u_w, logits_u_s = logits[2*batch_size:].chunk(chunks=2, dim=0)

        # semi-supervised loss
        Lu = (mask * self.loss_u(logits_u_s, p_targets_u, reduction='none')).mean()
        # supervised loss
        Lx = (self.loss_x(logits_x, targets_x, reduction='none') + self.loss_x(logits_inter, targets_x, reduction='none')).sum()/(2*batch_size)

        loss = Lx + self.cfg.lambda_u * Lu

        # calculate pseudo label acc
        pseudo_label_acc = self._get_pseudo_label_acc(p_targets_u, mask, targets_u)

        loss_dict = {
            "loss": loss.item(),
            "loss_x": Lx.item(),
            "loss_u": Lu.item(),
            "mask_prob": mask.float().mean().item(),
            "pseudo_acc": pseudo_label_acc.item(),
        }
        return loss, loss_dict
