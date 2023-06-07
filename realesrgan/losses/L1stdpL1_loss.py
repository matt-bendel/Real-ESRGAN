import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import l1_loss

@LOSS_REGISTRY.register()
class L1L1STDP(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1STDP, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.betastd = 1
        self.beta_1 = loss_weight
        self.beta_N = self.beta_1 * np.sqrt(1 / np.pi) + np.sqrt(2 / (np.pi * 2 * (2+ 1)))
        self.reduction = reduction

    def update_loss_weight(self, new_weight):
        self.betastd = new_weight

    def forward(self, gens, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        per_samp_l1 = 0
        for z in range(gens.shape[0]):
            per_samp_l1 += l1_loss(gens[z], target, weight, reduction=self.reduction)

        return per_samp_l1 / gens.shape[0] + l1_loss(torch.mean(gens, dim=0), target, weight, reduction=self.reduction) - self.betastd * self.beta_N * torch.std(gens, dim=0).mean()