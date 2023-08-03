import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
from mmcv.cnn import ConvModule
import torch
from mmseg.models import builder
import mmcv
import numpy as np
from mmseg.models.backbones.unet import BasicConvBlock
from ..utils import UpConvBlock
import torch.nn.functional as F
from torchvision.utils import save_image


@HEADS.register_module()
class DoubleGenerationHead0322(nn.Module):
    """Simplest depth prediction head, with only one fc layer.
    从 rgb 来预测 depth，注意这里的loss计算 即rgd
    从 depth 来预测 rgb，注意这里的loss计算 即dgr
    和double_generation_head_1217.py不同，这里用了unet来作为恢复和深度估计的辅助
    和0310相比，注意这里使用HHA作为输入，所以输入和输出都是三通道，
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 moco_out_channels=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_image_size=256,
                 edge_loss=None,
                 edge_weight=0.2,
                 l1_loss=None,
                 l1_loss_weight=None,
                 ssim_loss=None,
                 ssim_weight=0.84,
                 img_norm_cfg=None,
                 label_norm_cfg=None,
                 #####################
                 unethead_in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False
    ):
        super(DoubleGenerationHead0322, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.out_image_size = out_image_size
        # self.mseloss = nn.MSELoss()
        if l1_loss is not None:
            self.l1_loss = builder.build_loss(l1_loss)
            self.l1_loss_weight = l1_loss_weight
        else:
            self.l1_loss = None

        if edge_loss is not None:
            self.edge_loss = builder.build_loss(edge_loss)
            self.edge_weight = edge_weight
        else:
            self.edge_loss = None

        if ssim_loss is not None:
            self.ssim_loss = builder.build_loss(ssim_loss)
            self.ssim_weight = ssim_weight
        else:
            self.ssim_loss = None

        if img_norm_cfg is not None:
            self.img_norm_cfg = img_norm_cfg

        # 用于产生moco，按维度pooling
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.label_norm_cfg = label_norm_cfg
        # self.fc_cls = nn.Linear(in_channels, num_classes)
        convs = []
        self.norm_cfg = norm_cfg
        self.moco_out_channels = moco_out_channels
        # convs.append(
        #     ConvModule(
        #         self.in_channels,
        #         self.moco_out_channels,
        #         kernel_size=1,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=dict(type='ReLU')))
        #
        # self.convs = nn.Sequential(*convs)

        # self.rgd_linear64d = nn.Linear(3 * 64 * 64, 64)  # 输入 841维度，输出64维度，底下还有一个，分别对应rgb和dgr
        # self.rgd_convs_trans1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=(4, 4), stride=4)
        # self.rgd_convs_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=(2, 2), stride=2)

        self.rgd_convs = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        # self.rgd_convs_bn = nn.BatchNorm2d(1, affine=True)
        self.sigmoid = nn.Sigmoid()

        # self.dgr_linear64d = nn.Linear(3 * 64 * 64, 64)  # 输入 841维度，输出64维度
        # self.dgr_convs_trans1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=(4, 4), stride=4)
        # self.dgr_convs_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=(2, 2), stride=2)
        self.dgr_convs = nn.Conv2d(256, 3, kernel_size=1, stride=1)


        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.moco_out_channels))


        if unethead_in_channels is not None:
            self.num_stages = num_stages
            self.strides = strides
            self.downsamples = downsamples
            self.norm_eval = norm_eval
            self.base_channels = base_channels
            self.decoder = nn.ModuleList()

            self.init_unethead(
                base_channels,
                num_stages,
                strides,
                dec_num_convs,
                downsamples,
                enc_dilations,
                dec_dilations,
                with_cp,
                conv_cfg,
                norm_cfg,
                act_cfg,
                upsample_cfg)

    def init_unethead(self,
                      base_channels=64,
                      num_stages=5,
                      strides=(1, 1, 1, 1, 1),
                      dec_num_convs=(2, 2, 2, 2),
                      downsamples=(True, True, True, True),
                      enc_dilations=(1, 1, 1, 1, 1),
                      dec_dilations=(1, 1, 1, 1),
                      with_cp=False,
                      conv_cfg=None,
                      norm_cfg=dict(type='BN'),
                      act_cfg=dict(type='ReLU'),
                      upsample_cfg=dict(type='InterpConv'),
                      norm_eval=False,
                      dcn=None,
                      plugins=None,
                      pretrained=None,
                      init_cfg=None):
        for i in range(num_stages):
            # enc_conv_block = []
            if i != 0:
                # if strides[i] == 1 and downsamples[i - 1]:
                #     enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**(i + 2),
                        skip_channels=base_channels * 2**(i+1),
                        out_channels=base_channels * 2**(i+1),
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

    def rgd_forward(self, x, momentum=False):
        # if self.with_avg_pool:
        #     assert x.dim() == 4, \
        #         "Tensor must has 4 dims, got: {}".format(x.dim())
        # x = x[0:x.shape[0]//2]
        out_conv_trans1 = None
        if not momentum:
            out_conv_trans1 = self.rgd_convs(x)  # generate depth of image
            out_conv_trans1 = self.sigmoid(out_conv_trans1)
        # 这部分需要产生输出的
        # output = self.convs(x)
        # output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        # dep_head_out_for_moco_64d = self.avg_pool(output)
        if self.with_avg_pool:
            x = self.avg_pool(x)
        dep_head_out_for_moco_64d = self.mlp(x.view(x.size(0), -1))

        return out_conv_trans1, dep_head_out_for_moco_64d

    def unet_decoder_forward(self, x, momentum=None):
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

    def rgd_loss(self, dep_preds, labels, img_metas=None):
        '''
        :param dep_preds:
        :param labels: 这里的是经过normalization之后的数值
        :param img_metas:
        :return:
        '''

        # labels1 = labels.permute(1, 2, 0)
        losses = dict()
        resized_dep_preds = F.upsample_bilinear(dep_preds, size=[labels.shape[2], labels.shape[3]])
        # resized_labels = F.interpolate(labels, size=[self.out_image_size, self.out_image_size])

        # if self.edge_loss is not None:
        #     losses['loss_edge'] = self.edge_loss(dep_preds, labels) * (1 - self.edge_weight)
        # save_image(labels, "newback/imgs_segs/imgs_head/0323depth_targets_" + img_metas[0]['ann_info']['seg_map'])

        if self.ssim_loss is not None:
            # 窗口设置为3*3
            losses['loss_ssim_rgd'] = self.ssim_loss(resized_dep_preds * 255, labels * 255) * self.ssim_weight

        if self.l1_loss is not None:
            losses['loss_l1_rgd'] = self.l1_loss(resized_dep_preds, labels) * self.l1_loss_weight
        # save_image(resized_dep_preds,
        #            "newback/imgs_segs/rgd/depth_prediction_" + img_metas[0]['ori_filename'])  # 在这一步，*255，恢复图像信息
        # save_image(labels, "newback/imgs_segs/rgd/labels_" + img_metas[0]['ori_filename'])
        return losses

    def dgr_forward(self, x, momentum=False):
        out_conv_trans1 = None
        if not momentum:
            out_conv_trans1 = self.dgr_convs(x)
            out_conv_trans1 = self.sigmoid(out_conv_trans1)

        # output = self.convs(x)
        # # output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        # img_head_out_for_moco_64d = self.avg_pool(output)
        if self.with_avg_pool:
            x = self.avg_pool(x)
        # dep_head_out_for_moco_64d = self.mlp(x.view(x.size(0), -1))
        img_head_out_for_moco_64d = self.mlp(x.view(x.size(0), -1))
        return out_conv_trans1, img_head_out_for_moco_64d  # [img_prediction] if self.img_norm_cfg is not None else [output]

    def dgr_loss(self, img_preds, targets, img_metas=None):
        losses = dict()
        resized_targets = F.upsample_bilinear(img_preds, size=[targets.shape[2], targets.shape[3]])
        # resized_targets = F.interpolate(targets, size=[self.out_image_size, self.out_image_size])
        if self.ssim_loss is not None:
            losses['loss_ssim_dgr'] = self.ssim_loss(resized_targets * 255, targets * 255) * self.ssim_weight
        if self.l1_loss is not None:
            losses['loss_l1_dgr'] = self.l1_loss(resized_targets, targets) * self.l1_loss_weight

        # save_image(resized_targets, "newback/imgs_segs/dgr/img_prediction_" + img_metas[0]['ori_filename'])  # 在这一步，*255，恢复图像信息
        # save_image(targets, "newback/imgs_segs/dgr/targets_" + img_metas[0]['ori_filename'])
        return losses

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

    def resize_and_getmax(self, labels):
        labels = labels.to(torch.float32)
        resized = F.interpolate(labels, size=[self.out_image_size, self.out_image_size])
        max_value = torch.max(resized)
        return max_value, resized
