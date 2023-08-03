import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module()
class ContrastiveDecoupleHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 temperature=0.1,
                 contrastive_weight=0.1):
        super(ContrastiveDecoupleHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)


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
            losses['loss_'+flag+'_contrastive'] = self.criterion(logits, labels) * self.contrastive_weight  # float32转float64
        else:
            losses['loss_contrastive'] = self.criterion(logits, labels) * self.contrastive_weight  # float32转float64
        return losses
