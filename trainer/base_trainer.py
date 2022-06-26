""" 
This file set the base trainer format for the framework
To write your own SSL althorithm, please inherit base trainer as a base class
"""

import torch

class Trainer(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.threshold = cfg.get("threshold", 0.95)
        self.T = cfg.get("T", 1.0)

    def compute_loss(self):
        raise NotImplementedError

    def get_pseudo_acc(self, p_targets, targets, mask=None):
        if mask is not None:
            right_labels = (p_targets == targets).float() * mask
            return right_labels.sum() / max(mask.sum(), 1.0)
        else:
            right_labels = (p_targets == targets).float()
            return right_labels.mean()

    @torch.no_grad()
    def _get_pseudo_label_acc(self, p_targets_u, mask, targets_u):
        targets_u = targets_u.to(self.device)
        right_labels = (p_targets_u == targets_u).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)
        return pseudo_label_acc

    @torch.no_grad()
    def _get_psuedo_label_and_mask(self, probs_u_w):
        """
        max probs > thrs
        """
        max_probs, p_targets_u = torch.max(probs_u_w, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        return p_targets_u, mask