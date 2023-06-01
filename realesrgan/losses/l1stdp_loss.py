import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.basic_loss import l1_loss

@LOSS_REGISTRY.register()
class L1STDP(nn.Module):
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

        self.betastd = loss_weight
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
        return l1_loss(torch.mean(gens, dim=1), target, weight, reduction=self.reduction) - self.betastd * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()