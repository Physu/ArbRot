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
from mmcv.runner import BaseModule
import time


@HEADS.register_module()
class DepthGenerateRGBHead(BaseModule):
    """Simplest depth prediction head, with only one fc layer.
    从 rgb 来预测 depth，注意这里的loss计算 即rgd
    从 depth 来预测 rgb，注意这里的loss计算 即dgr
    和double_generation_head_1217.py不同，这里用了unet来作为恢复和深度估计的辅助
    和0310相比，注意这里使用HHA作为输入，所以输入和输出都是三通道，
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=256,
                 conv_channels=(128, 3),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 conv_cfg=None,
                 bias=None,
                 edge_loss=None,
                 edge_weight=0.2,
                 l1_loss=None,
                 l1_loss_weight=None,
                 ssim_loss=None,
                 ssim_weight=0.84,
                 loss_total=None,
                 loss_total_weight=None):
        super(DepthGenerateRGBHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        # self.mseloss = nn.MSELoss()
        # if l1_loss is not None:
        #     self.l1_loss = builder.build_loss(l1_loss)
        #     self.l1_loss_weight = l1_loss_weight
        # else:
        #     self.l1_loss = None

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

        self.loss_total = loss_total
        self.loss_total_weight = loss_total_weight

        # 用于产生moco，按维度pooling
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg
        self.bias = bias

        # self.rgd_convs = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        # # self.rgd_convs_bn = nn.BatchNorm2d(1, affine=True)
        # self.sigmoid = nn.Sigmoid()

        self.flag = None

        if len(self.conv_channels) > 0:
            self.dgr_convs = self._add_conv_branch(
                self.in_channels, self.conv_channels)

        self.img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            to_rgb=True)
        self.numerator = 0

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

    def forward(self, x, momentum=False):
        # if self.with_avg_pool:
        #     assert x.dim() == 4, \
        #         "Tensor must has 4 dims, got: {}".format(x.dim())
        # x = x[0:x.shape[0]//2]
        # out_conv_trans1 = None
        # if not momentum:
        #     out_conv_trans1 = self.rgd_convs(x)  # generate depth of image
        #     out_conv_trans1 = self.sigmoid(out_conv_trans1)
        # 这部分需要产生输出的
        # output = self.convs(x)
        # output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        # dep_head_out_for_moco_64d = self.avg_pool(output)
        if self.with_avg_pool:
            x = self.avg_pool(x)
        rgb = self.dgr_convs(x)

        return rgb

    def loss(self, rgb_preds, imgs, imgs_aug, img_metas=None):
        '''
        :param dep_preds:
        :param labels: 这里的是经过normalization之后的数值
        :param img_metas:
        :return:
        '''

        # labels1 = labels.permute(1, 2, 0)
        losses = dict()
        resized_rgb_preds = F.upsample_bilinear(rgb_preds, size=[imgs.shape[2], imgs.shape[3]])
        # resized_labels = F.interpolate(labels, size=[self.out_image_size, self.out_image_size])

        # if self.edge_loss is not None:
        #     losses['loss_edge'] = self.edge_loss(dep_preds, labels) * (1 - self.edge_weight)
        # save_image(labels, "newback/imgs_segs/imgs_head/0323depth_targets_" + img_metas[0]['ann_info']['seg_map'])

        if self.ssim_loss is not None:
            # 窗口设置为3*3
            losses['loss_ssim_dgr'] = self.ssim_loss(resized_rgb_preds * 255, imgs * 255) * self.ssim_weight

        # if self.l1_loss is not None:
        #     losses['loss_l1_dgr'] = self.l1_loss(resized_rgb_preds, imgs) * self.l1_loss_weight

        if self.loss_total is not None:
            tv_loss = self.TV_loss(resized_rgb_preds)
            style_loss = self.style_loss(imgs, resized_rgb_preds)
            preceptual_loss = self.preceptual_loss(imgs, resized_rgb_preds)
            valid_loss = self.l1_loss(resized_rgb_preds, imgs)
        if self.loss_total_weight is not None:
            losses['loss_dgr_total'] = (tv_loss * 0.1 + style_loss * 120 + preceptual_loss * 0.05 + valid_loss * 1) * self.loss_total_weight
        else:
            losses['loss_dgr_total'] = tv_loss * 0.1 + style_loss * 120 + preceptual_loss * 0.05 + valid_loss * 1



        ##################### 以后

        cuda_device = rgb_preds.get_device()  # 下面是为了计算计算模型大小和计算量的一个workaround
        if cuda_device != -1:
            mean = torch.tensor(self.img_norm_cfg['mean']).to(cuda_device)
            std = torch.tensor(self.img_norm_cfg['std']).to(cuda_device)
        else:
            mean = torch.tensor(self.img_norm_cfg['mean']).cpu()
            std = torch.tensor(self.img_norm_cfg['std']).cpu()
        # mean = torch.tensor(self.img_norm_cfg['mean']).cuda()
        mean = mean.reshape([len(mean), 1, 1]).repeat(1, 256, 256)
        # std = torch.tensor(self.img_norm_cfg['std']).cuda()
        std = std.reshape([len(std), 1, 1]).repeat(1, 256, 256)
        for i in range(len(resized_rgb_preds)):
            if img_metas[i]['ori_filename'] == '000147.jpg' or img_metas[i]['ori_filename'] == '000148.jpg' or img_metas[i]['ori_filename'] == '000149.jpg' or img_metas[i]['ori_filename'] == '000150.jpg':
                self.numerator = self.numerator + 1
                # save_image(torch.cat(((resized_rgb_preds[i]*std+mean).unsqueeze(0), (imgs[i]*std+mean).unsqueeze(0), (imgs_aug[0]*std+mean).unsqueeze(0)), dim=0),
                #            "newback/see_results/dep/1009only_rgd_dgr/1009_img_prediction_" + img_metas[i]['ori_filename'])
                save_image(torch.cat(((resized_rgb_preds[i] * std + mean).unsqueeze(0), (imgs[i] * std + mean).unsqueeze(0),
                                      (imgs_aug[i] * std + mean).unsqueeze(0)), dim=0),
                           "newback/see_results/img/20230706/1009_img_prediction_"+ str(self.numerator ) + img_metas[i]['ori_filename'])

                print("fuck, it is running!")


        # mean = torch.tensor(self.img_norm_cfg['mean']).cuda()
        # mean = mean.reshape([len(mean), 1, 1]).repeat(1, 256, 256)
        # std = torch.tensor(self.img_norm_cfg['std']).cuda()
        # std = std.reshape([len(std), 1, 1]).repeat(1, 256, 256)
        # rgb_preds_upsample = (F.interpolate(rgb_preds, size=(imgs.shape[2], imgs.shape[3]))*std + mean) * 255
        # imgs_recovery = (imgs*std+mean)*255
        #
        # for index in range(len(img_metas)):
        #     rgb_pd = rgb_preds_upsample[index].cpu().detach().numpy()
        #     img = imgs_recovery[index].cpu().numpy()
        #     if img_metas[index]['ori_filename'] == '008078.jpg':
        #         ori_img2_save = mmcv.imwrite(rgb_pd.transpose((1, 2, 0)),
        #                                      "newback/see_results/img_simsiam/1002_"+str(self.numerator)+"pre_" + img_metas[index]['ori_filename'])
        #         # ori_img2_save = mmcv.imwrite(img.transpose((1, 2, 0)),
        #         #                              "newback/see_results/img/1002_"+str(self.numerator)+"gt_" + img_metas[index]['ori_filename'])
        #         self.numerator = self.numerator + 1
            # print(img_metas[index]['ann_info']['seg_map'])


            # ori_img2_save = mmcv.imwrite(rgb_pd.transpose((1, 2, 0)),
            #                      "newback/see_results/img_simsiam/1001_imgpre_" + img_metas[index]['ori_filename'])
            # ori_img2_save = mmcv.imwrite(img.transpose((1, 2, 0)),
            #                              "newback/see_results/img_simsiam/1001_imggt_" + img_metas[index]['ori_filename'])
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

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            c, w, h = A_feat.size()
            # A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            # B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
