import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@LOSSES.register_module()
class EdgeAwareLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(EdgeAwareLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                depth,
                target_depth):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        grad_disp_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_disp_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(target_depth[:, :, :, :-1] - target_depth[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(target_depth[:, :, :-1, :] - target_depth[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()
