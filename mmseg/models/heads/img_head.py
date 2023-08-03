import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
from mmcv.cnn import ConvModule
import torch
from mmseg.models import builder
import cv2
import numpy as np
import mmcv


@HEADS.register_module()
class ImgHead(nn.Module):
    """Simplest img prediction head, with only one fc layer.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 out_channels=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_depth_image_size=225,
                 img_loss=dict(type='L1Loss'),
                 ssim_loss=None,
                 ssim_weight=0.84,  # from 'Loss Functions for Image Restoration with Neural Networks 2017'
                 edge_loss=None,
                 edge_weight=None,
                 img_norm_cfg=None):
        super(ImgHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.out_depth_image_size = out_depth_image_size
        self.l1loss = builder.build_loss(img_loss)

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc_cls = nn.Linear(in_channels, num_classes)
        convs = []
        self.norm_cfg = norm_cfg
        self.channels = out_channels
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        self.convs = nn.Sequential(*convs)

        self.img_norm_cfg = img_norm_cfg if img_norm_cfg is not None else None
        if ssim_loss is not None:
            self.ssim_loss = builder.build_loss(ssim_loss)
            self.ssim_weght = ssim_weight

        if edge_loss is not None:
            self.edge_loss = builder.build_loss(edge_loss)
            self.edge_weight = edge_weight if edge_weight is not None else (1 - ssim_weight)

        self.linear64d = nn.Linear(3 * 29 * 29, 64)  # 输入 841维度，输出64维度
        # self.convtrans2d = nn.ConvTranspose2d(3, 3, 3, stride=1, padding=0, output_padding=0, groups=1,
        #                  bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)


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

    def im_unnormalize(self, img, mean, std, to_rgb=True):
        """Inplace unnormalize an image with mean and std.
        根据给定的均值和方差，对图片进行恢复，为后面计算loss做准备

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        mean_cuda = torch.tensor(mean, dtype=torch.float32).cuda()
        mean_cuda = torch.cat((mean_cuda[0].repeat(2, 1, 225, 225),
                               mean_cuda[1].repeat(2, 1, 225, 225),
                               mean_cuda[2].repeat(2, 1, 225, 225)), dim=1)

        std_cuda = torch.tensor(std, dtype=torch.float32).cuda()
        std_cuda = torch.cat((std_cuda[0].repeat(2, 1, 225, 225),
                              std_cuda[1].repeat(2, 1, 225, 225),
                              std_cuda[2].repeat(2, 1, 225, 225)), dim=1)

        img_cuda = torch.mul(img, std_cuda)
        img_cuda = torch.add(img_cuda, mean_cuda)
        # img_cuda = img_cuda.to(torch.int32)
        img_cuda = torch.clamp(img_cuda, min=0, max=255)

        return img_cuda

    def forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())

        output = self.convs(x)
        output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        img_head_out_for_moco_64d = self.linear64d(output_flatten)
        m = torch.nn.Upsample(size=self.out_depth_image_size)
        output = m(output)
        #
        # cls_score = self.fc_cls(x)
        return output, img_head_out_for_moco_64d  #[img_prediction] if self.img_norm_cfg is not None else [output]

    def loss(self, img_preds, targets, img_metas=None):

        img_preds = img_preds[0]
        if self.img_norm_cfg is not None:
            img_prediction = self.im_unnormalize(img_preds,
                                                 self.img_norm_cfg['mean'],
                                                 self.img_norm_cfg['std'],
                                                 self.img_norm_cfg['to_rgb'])

            targets = self.im_unnormalize(targets,
                                          self.img_norm_cfg['mean'],
                                          self.img_norm_cfg['std'],
                                          self.img_norm_cfg['to_rgb'])

            ori_img1_save = mmcv.imwrite(img_prediction[0].to(torch.int32).cpu().numpy().transpose((1, 2, 0)),
                                         "newback/imgs_segs/imgs_head/img_prediction_" + img_metas[0]['ori_filename'])
            ori_img2_save = mmcv.imwrite(targets[0].to(torch.int32).cpu().numpy().transpose((1, 2, 0)),
                                         "newback/imgs_segs/imgs_head/targets_" + img_metas[1]['ori_filename'])

        losses = dict()
        losses['loss_img'] = self.l1loss(img_prediction, targets) * (1 - self.ssim_weght)
        if self.ssim_loss is not None:
            losses['loss_ssim'] = self.ssim_loss(img_prediction, targets) * self.ssim_weght
        #     # losses.update(loss_loc_depth_q)
        # if self.edge_loss is not None:
        #     losses['loss_edge'] = self.edge_loss(cls_score[0], labels)

        return losses
