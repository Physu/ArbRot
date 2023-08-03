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
class RotHead0403(BaseModule):
    """Simplest classifier head, with only one fc layer.
    不采取之前的360类别，采用12个大类，然后再通过回归的方式得到精确的角度

    准备接入U-Net的输出，来作为一个旋转的输出处理
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_rot_bins=24,
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
        super(RotHead0403, self).__init__()
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
        rot_preds_trans = rot_pred.view(rot_pred.size(0), -1)

        end += int(self.num_rot_bins / 2)
        results['rot_class'] = rot_preds_trans[..., start:end].contiguous()

        start = end
        end += int(self.num_rot_bins / 2)
        rot_res_norm = rot_preds_trans[..., start:end].contiguous()
        # print(f"rot_res_norm: {rot_res_norm}")
        rot_res_norm = torch.clamp(rot_res_norm, min=-1.0, max=1.0)
        results['rot_res_norm'] = rot_res_norm  # 这个用来计算loss
        # print(f"rot_res_norm_after: {rot_res_norm}")
        # results['rot_res'] = rot_res_norm * (np.pi / self.num_rot_bins)

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
        angle_per_class = 2 * np.pi / float(self.num_rot_bins/2)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)  # 这里为何要加15度，是为了 15, 45, 75, ..., 相当于将角度修正15度,也就是0度其实是cls 0 的中心位置
        angle_cls = shifted_angle // angle_per_class  # 整数除法返回向下取整后的结果
        angle_res = shifted_angle - (
                angle_cls * angle_per_class + angle_per_class / 2)  # 也就是距离这个ang-cls的弧度距离，如果为负，表示距离这个angle-cls差一点，为正，表示其值超过这个angle-cls 一点
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=False):
        """Inverse function to angle2class.

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / (float(self.num_rot_bins) / 2)  # 30 为一个间隔
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

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
        (rot_class_target, rot_res_target) = self.angle2class(rot_labels)

        rot_res_target /= self.norm_angle  # 归一化到 -0.5, +0.5 之间 ，相对角度
        # 0414 修改了 rotation loss的计算方式，去除无用loss
        dir_class_targets = rot_labels
        box_loss_weights = 1 / (float(self.num_rot_bins/2) + 1e-6)

        batch_size = dir_class_targets.shape[0]  # 注意这个batchsize是实际的二倍
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, self.num_rot_bins//2))

        heading_label_one_hot.scatter_(1, rot_class_target.unsqueeze(-1), 1)  # 注意这里，第一个1 表示需要修改的维度，第二个表示需要修改数据的index，第三个表示需要插入的数据
        heading_res_loss_weight = heading_label_one_hot * box_loss_weights

        # box_loss_weights = 1.0
        # heading_loss_weights = 1.0
        # heading_res_loss_weight = 1.0

        # calculate rotection class loss, CrossEntropyLoss
        losses['rot_class_loss'] = self.loss_total_weight * self.rot_class_loss(
            results['rot_class'],
            rot_class_target) / (batch_size + 1e-6)

        # calculate rotection residual loss, SmoothL1
        # 如果是这种处理，那就是让所有的res 都尽力靠近正确res
        # masked = torch.zeros(size=results['rot_class'].shape).cuda()
        #
        # for i, index in enumerate(rot_class_target):
        #     masked[i][index] = 1.
        # losses['rot_res_loss'] = self.rot_res_loss(
        #     results['rot_res_norm'] * masked,
        #     rot_res_target.unsqueeze(-1).repeat(1, int(self.num_rot_bins/2)) * masked) / (rot_class_target.shape[0] + 1e-8)

        losses['rot_res_loss'] = self.loss_total_weight * self.rot_res_loss(
            results['rot_res_norm'],
            rot_res_target.unsqueeze(-1).repeat(1, int(self.num_rot_bins/2)), heading_res_loss_weight)

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
        (rot_class_target, rot_res_target) = self.angle2class(rot_labels)
        rot_res_target /= self.norm_angle  # 归一化到 -0.5, +0.5 之间 ，相对角度
        # 关键注意这里，rot_res_target 是有一个normalization操作，这个得注意
        # restore_angle = self.class2angle(rot_class_target, rot_res_target*self.norm_angle)

        # dir_class_targets = rot_labels
        # box_loss_weights = 1 / (float(self.num_rot_bins / 2) + 1e-6)

        # batch_size = dir_class_targets.shape[0]  # 注意这个batchsize是实际的二倍
        # heading_label_one_hot = dir_class_targets.new_zeros(
        #     (batch_size, self.num_rot_bins // 2))

        # heading_label_one_hot.scatter_(1, rot_class_target.unsqueeze(-1),
        #                                1)  # 注意这里，第一个1 表示需要修改的维度，第二个表示需要修改数据的index，第三个表示需要插入的数据
        # heading_res_loss_weight = heading_label_one_hot * box_loss_weights

        rot_cls_pred, rot_cls_pred_index = torch.max(results['rot_class'], 1)
        rot_res_pred = results['rot_res_norm']
        restore_angle = self.class2angle(rot_cls_pred_index[0], rot_res_pred[0][rot_cls_pred_index] * self.norm_angle)

        restore_rotation = restore_angle[0] / np.pi * 180
        # losses['rot_cls_pred'] = rot_cls_pred_index
        # losses['rot_res_pred'] = rot_res_pred
        # losses['restore_angle'] = restore_angle

        if rot_cls_pred_index[0] == rot_class_target[0]:
            losses['rot_class_correct'] = True
        else:
            losses['rot_class_correct'] = False
        rotation_res_gap = torch.abs(rot_res_pred[0][rot_cls_pred_index] - rot_res_target[0])
        rotation_gap = torch.abs(restore_rotation - rot_labels)
        losses['rotation_gap'] = rotation_gap
        losses['rotation_res_gap'] = rotation_res_gap

        return losses