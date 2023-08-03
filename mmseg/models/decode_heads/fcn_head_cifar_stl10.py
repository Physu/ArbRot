import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..builder import build_loss
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..losses import accuracy


@HEADS.register_module()
class FCNHeadCifar10STL10(BaseDecodeHead):
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
                 num_classification=None,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHeadCifar10STL10, self).__init__(**kwargs)

        self.num_classification = num_classification
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
        self.classification_prediction = ConvModule(
                2048,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                bias=True)

        self.fc = nn.Linear(self.num_classes * 16, self.num_classes)  # 之所以*16，是因为最后特征尺寸4*4

    def forward(self, inputs):
        """Forward function."""
        # 这里部分的处理
        x = self._transform_inputs(inputs)  # [[B,1024,29,29],[B,2048,29,29]]  对于cifar10 [[B,1024,82,82],[B,2048,82,82]]
        output = self.convs(x[0])
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        output = output.view(output.shape[0], -1)  # flatten 操作
        output = self.fc(output)  # 最后返回[B,11] 表示对应11个类别的score

        return output


    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = torch.tensor([seg_label[0][0][0][0], seg_label[1][0][0][0]]).cuda()

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

