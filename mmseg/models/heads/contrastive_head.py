import torch
import torch.nn as nn

from ..registry import HEADS
from mmcv.runner import BaseModule


@HEADS.register_module()
class ContrastiveHead(BaseModule):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
                 temperature=0.1,
                 loss_total_weight=1.0):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.loss_total_weight = loss_total_weight

    def forward(self, pos, neg, flag=None):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        if flag:  # 添加可能存在的标签信息，用来区分
            losses['loss_'+flag+'_contrastive'] = self.criterion(logits, labels) * self.loss_total_weight  # float32转float64
        else:
            losses['loss_contrastive'] = self.criterion(logits, labels) * self.loss_total_weight  # float32转float64
        return losses
