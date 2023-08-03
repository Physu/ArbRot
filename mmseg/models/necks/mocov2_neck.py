# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS
from mmseg.models import builder_moco
from mmseg.core import add_prefix
import torch


@NECKS.register_module()
class MoCoV2Neck(BaseModule):
    """The non-linear neck of MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 rot_and_loc_head=None,
                 double_generation_head=None,
                 with_avg_pool=True,
                 init_cfg=None):
        super(MoCoV2Neck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
        #     nn.Linear(hid_channels, out_channels))

        if double_generation_head is not None:
            self.double_generation_head = builder_moco.build_head_moco(double_generation_head)
        else:
            self.double_generation_head = None

        if rot_and_loc_head is not None:
            self.rot_and_loc_head = builder_moco.build_head_moco(rot_and_loc_head)
        else:
            self.rot_and_loc_head = None

    def double_generation_head_forward_train_v2(self,
                                             combination_img_and_gt,
                                             img_concate=None,
                                             gt_semantic_seg_concate=None,
                                             img_metas=None,
                                             return_loss=True,
                                             momentum=False):
        '''
        :param combination_img_and_gt: input is composed of img and depth
        :param img_concate:
        :param gt_semantic_seg_concate:
        :param img_metas:
        :param return_loss:
        :param momentum: whether is momentum branch, in this situation, no need to recover the image and depth
        :return:
        '''
        losses = dict()
        total_128d = None
        # ptr = combination_img_and_gt.shape[0]
        unet_decoder_out = self.double_generation_head.unet_decoder_forward(
            combination_img_and_gt,
            momentum)
        if not momentum:  # 判断是来自gradient update or momentum update, 下面执行的是 gradient update
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(unet_decoder_out[3][0:img_concate.shape[0]], momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(unet_decoder_out[3][img_concate.shape[0]:], momentum)
            if return_loss and not momentum:
                loss_rgd = self.double_generation_head.rgd_loss(out_rgd, gt_semantic_seg_concate,
                                                                img_metas)
                losses.update(add_prefix(loss_rgd, "rgd"))

                loss_dgr = self.double_generation_head.dgr_loss(out_dgr, img_concate, img_metas)
                losses.update(add_prefix(loss_dgr, "dgr"))
        else:
            # 用来输出需要的64d的特征向量
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(
                unet_decoder_out[3][0:img_concate.shape[0]], momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(
                unet_decoder_out[3][img_concate.shape[0]:], momentum)

        if out_dgr_for_moco_64d is not None and out_rgd_for_moco_64d is not None:
            total_128d = torch.cat((out_rgd_for_moco_64d, out_dgr_for_moco_64d), dim=1)

        return losses, total_128d

    def _rot_and_loc_head_forward_train(self, x, rot, loc, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.rot_and_loc_head.forward(x)
        loss = self.rot_and_loc_head.loss(out, rot, loc)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def forward(self, x, sunrgbd_rotation1=None, paste_location1=None, img_aug1=None, gt_semantic1=None, img_metas=None, momentum=None):
        '''

        :param x:
        :param sunrgbd_rotation1:
        :param paste_location1:
        :param img_aug1:
        :param gt_semantic1:
        :param img_metas:
        :param momentum: 用来区分是否是gradient update 或者 momentum update
        :return:
        '''
        if len(x) == 1:
            x = x[0]
        backbone = x[3]
        # if self.with_avg_pool:
        #     x = self.avgpool(x)
        '''below is modified'''
        losses = dict()

        if self.rot_and_loc_head is not None and momentum is None:
            # 之所以这样拼接是为了充分利用img和depth信息来做旋转和位置的预测
            backbone = torch.cat((backbone[0:backbone.shape[0] // 2], backbone[backbone.shape[0] // 2:]), dim=1)
            loss_rot_and_loc_rgb = self._rot_and_loc_head_forward_train(
                backbone,
                sunrgbd_rotation1.to(torch.int64),
                paste_location1.to(torch.int64), "rot_loc")
            losses.update(loss_rot_and_loc_rgb)

        if self.double_generation_head is not None:
            loss_gradient, q = self.double_generation_head_forward_train_v2(
                x,  # 这个x，应该市tuple, length 为4
                img_aug1,
                gt_semantic1,  # 注意这里，就是batchsize数目，没有concate操作，即未翻倍
                img_metas,
                momentum=momentum
            )

            losses.update(loss_gradient)

        # return [self.mlp(x.view(x.size(0), -1))], q, losses
        return q, losses