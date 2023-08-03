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


@HEADS.register_module()
class RotHead1216(nn.Module):
    """Simplest classifier head, with only one fc layer.
    不采取之前的360类别，采用12个大类，然后再通过回归的方式得到精确的角度
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_dir_bins=12,
                 rot_weight=0.2,
                 location_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 shared_conv_channels=(),
                 loc_conv_channels=(),
                 num_loc_out_channels=0,
                 dir_conv_channels=(),
                 num_dir_out_channels= 0,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 ):
        super(RotHead1216, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_dir_bins = num_dir_bins
        ####################################
        assert in_channels > 0
        assert num_loc_out_channels > 0
        assert num_dir_out_channels > 0
        self.in_channels = in_channels  # 1536
        self.shared_conv_channels = shared_conv_channels  # (512,128)
        self.loc_conv_channels = loc_conv_channels  # 128
        self.num_loc_out_channels = num_loc_out_channels  # 1 for car
        self.dir_conv_channels = dir_conv_channels  # 128
        self.num_dir_out_channels = num_dir_out_channels  # 30
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias
        if location_loss is not None:
            self.location_loss = builder.build_loss(location_loss)

        if dir_class_loss is not None:
            self.dir_class_loss = builder.build_loss(dir_class_loss)

        if dir_res_loss is not None:
            self.dir_res_loss = builder.build_loss(dir_res_loss)

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc_cls = nn.Linear(in_channels, num_dir_bins)
        # self.rot_weight = rot_weight

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # add cls specific branch
        prev_channel = out_channels
        if len(self.loc_conv_channels) > 0:
            self.loc_convs = self._add_conv_branch(prev_channel,
                                                   self.loc_conv_channels)
            prev_channel = self.loc_conv_channels[-1]

        self.conv_loc = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_loc_out_channels,
            kernel_size=1)
        # add reg specific branch
        prev_channel = out_channels
        if len(self.dir_conv_channels) > 0:
            self.dir_convs = self._add_conv_branch(prev_channel,
                                                   self.dir_conv_channels)
            prev_channel = self.dir_conv_channels[-1]

        self.conv_dir = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=num_dir_out_channels,
            kernel_size=1)

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

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (2)
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

    def forward(self, x):
        """Forward.

                Args:
                    feats (Tensor): Input features

                Returns:
                    Tensor: Class scores predictions
                    Tensor: Regression predictions
                """
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), x.size(1), -1)
        results = {}
        start, end = 0, 0
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(x)  # batchsize 必须大于1

        # separate branches
        x_loc = x
        x_dir = x

        if len(self.loc_conv_channels) > 0:
            x_loc = self.loc_convs(x_loc)
        loc_score = self.conv_loc(x_loc)
        loc_score = loc_score.view(loc_score.size(0), -1)

        results['loc_score'] = loc_score

        if len(self.dir_conv_channels) > 0:
            x_dir = self.dir_convs(x_dir)
        dir_pred = self.conv_dir(x_dir)

        # decode direction
        # dir_preds_trans = dir_pred.transpose(2, 1)
        dir_preds_trans = dir_pred.view(dir_pred.size(0), -1)

        end += int(self.num_dir_bins / 2)
        results['dir_class'] = dir_preds_trans[..., start:end].contiguous()
        start = end

        end += int(self.num_dir_bins / 2)
        dir_res_norm = dir_preds_trans[..., start:end].contiguous()

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (np.pi / self.num_dir_bins)

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


    def angle2class(self, angle):
        """Convert continuous angle to a discrete class and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle / 180. * np.pi
        angle = angle % (2 * np.pi)  # 这一步关键，这一步相当于做了一个修正，将所有的angle修正到0-2pi之间
        angle_per_class = 2 * np.pi / float(self.num_dir_bins/2)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)  # 这里为何要加15度，是为了 15, 45, 75, ..., 相当于将角度修正15度
        angle_cls = shifted_angle // angle_per_class  # 整数除法返回向下取整后的结果
        angle_res = shifted_angle - (
                angle_cls * angle_per_class + angle_per_class / 2)  # 也就是距离这个ang-cls的弧度距离，如果为负，表示距离这个angle-cls差一点，为正，表示其值超过这个angle-cls 一点
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class.

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def loss(self, results, rot_labels, loc_labels):
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
        (dir_class_target, dir_res_target) = self.angle2class(rot_labels)
        dir_res_target /= (2 * np.pi / self.num_dir_bins)  # 归一化到 -0.5, +0.5 之间 ，相对角度

        # box_loss_weights = 1.0
        # heading_loss_weights = 1.0
        # heading_res_loss_weight = 1.0

        # calculate location class loss, CrossEntropyLoss
        losses['location_loss'] = self.location_loss(
            results['loc_score'],
            loc_labels)
            # weight=box_loss_weights)  # 关于这个weights，必须和类别数目相匹配

        # calculate direction class loss, CrossEntropyLoss
        losses['dir_class_loss'] = self.dir_class_loss(
            results['dir_class'],
            dir_class_target)
            # weight=heading_loss_weights)

        # calculate direction residual loss, SmoothL1
        # 如果是这种处理，那就是让所有的res 都尽力靠近正确res
        losses['dir_res_loss'] = self.dir_res_loss(
            results['dir_res_norm'],
            dir_res_target.unsqueeze(-1).repeat(1, int(self.num_dir_bins/2)))
            # weight=heading_res_loss_weight)



        return losses
