import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
# from .decode_head import BaseDecodeHead
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..losses import accuracy


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHeadClassification(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPHeadClassification, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations  # 如论文中所示
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 这一步，将(B, channels, m, n) 适应性avg pool为 (B, channels, 1, 1)
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.fc = nn.Linear(self.num_classes * 16, self.num_classes)  # 之所以*16，是因为最后特征尺寸4*4

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]  # 这一步，将resnet1c的输出经过avgpooling, 然后conv2d, 再重新插值成需要的尺寸
        aspp_outs.extend(self.aspp_modules(x))  # 构建了一个特征尺寸金字塔，aspp_modules 会有四个不同dilation的 DCN, 不过输出的尺寸都是一致的
        aspp_outs = torch.cat(aspp_outs, dim=1)  # 1+4 最后将5层特征进行拼接
        output = self.bottleneck(aspp_outs)  # 处理拼接好的特征
        output = self.cls_seg(output)  # [B, classes, w , h]，这个wh还是一个缩略图的pixel classification，不是全尺寸，需要进一步处理
        output = output.view(output.shape[0], -1)  # flatten 操作
        output = self.fc(output)   # 最后返回[B,11] 表示对应11个类别的score

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
