import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
import torch
import numpy as np
from mmseg.models import builder
import math
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_conv_layer
from mmcv.runner import BaseModule


@HEADS.register_module()
class RotHeadFour0802(BaseModule):
    """Simplest classifier head, with only one fc layer.
    不采取之前的360类别，采用12个大类，然后再通过回归的方式得到精确的角度

    准备接入U-Net的输出，来作为一个旋转的输出处理
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_rot_bins=4,
                 rot_class_loss=None,
                 rot_res_loss=None,
                 shared_conv_channels=(),
                 num_rot_out_channels=0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 loss_total_weight=1.0
):
        super(RotHeadFour0802, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_rot_bins = num_rot_bins
        ####################################
        assert in_channels > 0
        assert num_rot_out_channels > 0
        self.in_channels = in_channels  # 1536
        self.shared_conv_channels = shared_conv_channels  # (512,128)
        self.num_rot_out_channels = num_rot_out_channels  # 30
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias
        self.norm_angle = (2 * np.pi / self.num_rot_bins)


        if rot_class_loss is not None:
            self.rot_class_loss = builder.build_loss(rot_class_loss)

        if rot_res_loss is not None:
            self.rot_res_loss = builder.build_loss(rot_res_loss)

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels
        prev_channel = out_channels

        self.conv_rot = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_rot_out_channels,
            kernel_size=1)

        self.loss_total_weight = loss_total_weight

    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        '''
        Attention:
            a = [13]
            b = [14]
            a + b  # [13,14] 注意这个list加法的处理
        '''
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (2)
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_rot_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_rot_bins * 2 + self.num_sizes * 4

    def forward(self, x):
        """Forward.

                Args:
                    feats (Tensor): Input features

                Returns:
                    Tensor: Class scores predictions
                    Tensor: Regression predictions
                """
        if self.with_avg_pool:  # x [B,512,32,32]
            x = self.avg_pool(x[3])
        x = x.view(x.size(0), x.size(1), -1)  # x [B,512,1]
        results = {}
        start, end = 0, 0
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(x)  # batchsize 必须大于1 [2048->128]

        # separate branches

        x_rot = x

        rot_pred = self.conv_rot(x_rot)

        # decode rotection
        # rot_preds_trans = rot_pred.transpose(2, 1)
        results['rot_class'] = rot_pred.view(rot_pred.size(0), -1)

        return results

    def generate_bce_weight(self, x):
        '''
        用来生成所需要的weight信息，因为要预测360°的结果，所以设计有正负2度的容忍度
        :param x:
        :param mu: mean value
        :param sigma: std value
        :return:
        '''
        pdf = 1 - 1 / ((torch.exp(-x)/2)+(torch.exp(x)/2))
        return pdf

    def loss(self, results, rot_labels):
        '''
        import torch
        a = torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
        b = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]])
        tar = torch.tensor([4])
        l11 = self.criterion(a, tar)
        l22 = self.criterion(b, tar)
        注意这里，l11 和 l22 的数值是一样的，这个对于预测角度来说肯定是不合理，对于a，这种预测结果，应该
        要比b这种预测结果的loss 更小一些，而不是二者的loss相同
        :param cls_score:
        :param labels:
        :return:
        '''
        losses = dict()
        # mmdet3d/core/bbox/coders/partial_bin_based_bbox_coder.py 这个angle+15//2pi
        # 0802
        # 修改了 rotation loss的计算方式，去除无用loss
        batch_size = rot_labels.shape[0]  # 注意这个batchsize是实际的二倍
        # calculate rotection class loss, CrossEntropyLoss
        losses['rot_class_loss'] = self.loss_total_weight * self.rot_class_loss(results['rot_class'], rot_labels) / (batch_size + 1e-6)

        return losses

    def rot_evaluation(self, results, rot_labels):
        '''
        import torch
        a = torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
        b = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]])
        tar = torch.tensor([4])
        l11 = self.criterion(a, tar)
        l22 = self.criterion(b, tar)
        注意这里，l11 和 l22 的数值是一样的，这个对于预测角度来说肯定是不合理，对于a，这种预测结果，应该
        要比b这种预测结果的loss 更小一些，而不是二者的loss相同
        :param cls_score:
        :param labels:
        :return:
        '''
        losses = dict()
        # mmdet3d/core/bbox/coders/partial_bin_based_bbox_coder.py 这个angle+15//2pi


        rot_cls_pred, rot_cls_pred_index = torch.max(results['rot_class'], 1)

        if rot_cls_pred_index[0] == rot_labels[0]:
            losses['rot_class_correct'] = True
        else:
            losses['rot_class_correct'] = False
        return losses