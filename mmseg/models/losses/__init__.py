from .accuracy import Accuracy, accuracy
from .cls_accuracy import cls_Accuracy, cls_accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .cls_cross_entropy_loss import (ClsCrossEntropyLoss, cls_binary_cross_entropy,
                                     cls_cross_entropy, cls_soft_cross_entropy)
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from .mse_loss import MSELoss, mse_loss
from .direction_aware_ssim_loss import DirectionAwareSSIM_Loss
from .edge_aware_loss import EdgeAwareLoss
from .gaussian_focal_loss import GaussianFocalLoss
from .AutomaticWeightedLoss import AutomaticWeightedLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'L1Loss', 'l1_loss', 'smooth_l1_loss', 'SmoothL1Loss', 'MSELoss',
    'mse_loss', 'cls_Accuracy', 'cls_accuracy', 'ClsCrossEntropyLoss',
    'cls_soft_cross_entropy', 'cls_cross_entropy', 'cls_binary_cross_entropy',
    'DirectionAwareSSIM_Loss', 'EdgeAwareLoss', 'GaussianFocalLoss',
    'AutomaticWeightedLoss'
]
