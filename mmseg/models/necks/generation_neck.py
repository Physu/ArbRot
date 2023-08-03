# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..registry import NECKS
from mmseg.models import builder_moco
from mmseg.core import add_prefix
import torch

from mmseg.models.backbones.unet import BasicConvBlock
from ..utils import UpConvBlock


@NECKS.register_module()
class GenerationNeck(BaseModule):
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
                 unethead_in_channels=3,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=False),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 with_avg_pool=True,
                 init_cfg=None):
        super(GenerationNeck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            # enc_conv_block = []
            if i != 0:
                # if strides[i] == 1 and downsamples[i - 1]:
                #     enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2 ** (i + 2),
                        skip_channels=base_channels * 2 ** (i + 1),
                        out_channels=base_channels * 2 ** (i + 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

        self.flag = None

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
        # if self.with_avg_pool:
        #     x = self.avgpool(x)
        '''below is modified'''
        enc_outs = x
        x = x[3]
        self._check_input_divisible(x)

        # for enc in self.encoder:
        #     x = enc(x)
        #     enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'