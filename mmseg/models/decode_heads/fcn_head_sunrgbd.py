import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head_sunrgbd import BaseDecodeHeadSUNRGBD
from ..builder import build_loss
from mmseg.ops import resize
from ..losses import accuracy


@HEADS.register_module()
class FCNHeadSUNRGBD(BaseDecodeHeadSUNRGBD):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 loss_rotation=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.4),
                 loss_location=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.4),
                 num_angle=4,
                 num_location=9,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHeadSUNRGBD, self).__init__(**kwargs)
        self.loss_rotation = build_loss(loss_rotation)  # 这部分loss新定义
        self.loss_location = build_loss(loss_location)
        self.num_angle = num_angle
        self.num_location = num_location
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels[0],
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels[0] + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        norm_cfg = dict(type='BN2d', eps=1e-3, momentum=0.01)
        self.rotataion_prediction = ConvModule(
                2048,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                bias=True)

        self.rotation_linear = nn.Linear(29 * 29, self.num_angle, bias=False)
        self.location_linear = nn.Linear(29 * 29, self.num_location, bias=False)



    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)  # [[B,1024,29,29],[B,2048,29,29]]
        # 通过后面的处理可以看出，其实就是用x[0]
        output = []
        # output = self.convs(x[0])
        # if self.concat_input:
        #     output = self.conv_cat(torch.cat([x, output], dim=1))
        # output = self.cls_seg(output)

        output2 = self.rotataion_prediction(x[1])
        rotation_pred = self.rotation_linear(output2.squeeze(1).reshape(-1, 29*29))
        location_pred = self.location_linear(output2.squeeze(1).reshape(-1, 29*29))

        # return output, rotation_pred, location_pred
        return output, rotation_pred, location_pred