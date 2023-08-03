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


@HEADS.register_module()
class DoubleGenerationHead0308(nn.Module):
    """Simplest depth prediction head, with only one fc layer.
    从 rgb 来预测 depth，注意这里的loss计算 即rgd
    从 depth 来预测 rgb，注意这里的loss计算 即dgr
    和double_generation_head_1217.py不同，这里用了unet来作为恢复和深度估计的辅助
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 out_channels=3,
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
        super(DoubleGenerationHead0308, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.out_image_size = out_image_size
        # self.mseloss = nn.MSELoss()
        if l1_loss is not None:
            self.l1_loss = builder.build_loss(l1_loss)
            self.l1_loss_weight = l1_loss_weight

        if edge_loss is not None:
            self.edge_loss = builder.build_loss(edge_loss)
            self.edge_weight = edge_weight
        else:
            self.edge_loss = None

        if ssim_loss is not None:
            self.ssim_loss = builder.build_loss(ssim_loss)
            self.ssim_weght = ssim_weight

        if img_norm_cfg is not None:
            self.img_norm_cfg = img_norm_cfg
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.label_norm_cfg = label_norm_cfg
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
                act_cfg=dict(type='ReLU')))

        self.convs = nn.Sequential(*convs)

        self.rgd_linear64d = nn.Linear(3 * 64 * 64, 64)  # 输入 841维度，输出64维度，底下还有一个，分别对应rgb和dgr
        # self.rgd_convs_trans1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=(4, 4), stride=4)
        # self.rgd_convs_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=(2, 2), stride=2)

        self.rgd_convs = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.rgd_convs_bn = nn.BatchNorm2d(1, affine=True)
        # self.sigmoid = nn.Sigmoid()

        self.dgr_linear64d = nn.Linear(3 * 64 * 64, 64)  # 输入 841维度，输出64维度
        # self.dgr_convs_trans1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=(4, 4), stride=4)
        # self.dgr_convs_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=(2, 2), stride=2)
        self.dgr_convs = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.dgr_convs_bn = nn.BatchNorm2d(3, affine=True)


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

    def depth_unnormalize(self, img, mean, std):
        """Inplace unnormalize an image with mean and std.
        根据给定的均值和方差，对深度图进行恢复，为后面计算loss做准备

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.


        Returns:
            ndarray: The normalized image.
        """
        mean_cuda = torch.tensor(mean, dtype=torch.float32).cuda()
        mean_cuda = mean_cuda[0].repeat(img.size(0), 1, self.out_image_size, self.out_image_size)

        std_cuda = torch.tensor(std, dtype=torch.float32).cuda()
        std_cuda = std_cuda[0].repeat(img.size(0), 1, self.out_image_size, self.out_image_size)

        img_cuda = torch.mul(img, std_cuda)
        img_cuda = torch.add(img_cuda, mean_cuda)
        # img_cuda = img_cuda.to(torch.int32)
        img_cuda = torch.clamp(img_cuda, min=0)

        return img_cuda

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
        mean_cuda = torch.cat((mean_cuda[0].repeat(img.size(0), 1, self.out_image_size, self.out_image_size),
                               mean_cuda[1].repeat(img.size(0), 1, self.out_image_size, self.out_image_size),
                               mean_cuda[2].repeat(img.size(0), 1, self.out_image_size, self.out_image_size)), dim=1)

        std_cuda = torch.tensor(std, dtype=torch.float32).cuda()
        std_cuda = torch.cat((std_cuda[0].repeat(img.size(0), 1, self.out_image_size, self.out_image_size),
                              std_cuda[1].repeat(img.size(0), 1, self.out_image_size, self.out_image_size),
                              std_cuda[2].repeat(img.size(0), 1, self.out_image_size, self.out_image_size)), dim=1)

        img_cuda = torch.mul(img, std_cuda)
        img_cuda = torch.add(img_cuda, mean_cuda)
        # img_cuda = img_cuda.to(torch.int32)
        img_cuda = torch.clamp(img_cuda, min=0, max=255)

        return img_cuda


    def rgd_forward(self, x, momentum=False):
        # if self.with_avg_pool:
        #     assert x.dim() == 4, \
        #         "Tensor must has 4 dims, got: {}".format(x.dim())
        # x = x[0:x.shape[0]//2]
        out_conv_trans1 = None
        if not momentum:
            out_conv_trans1 = self.rgd_convs(x)  # generate depth of image
            out_conv_trans1 = self.rgd_convs_bn(out_conv_trans1)
        # 这部分需要产生输出的
        output = self.convs(x)
        output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        dep_head_out_for_moco_64d = self.rgd_linear64d(output_flatten)

        return out_conv_trans1, dep_head_out_for_moco_64d

    def unet_decoder_forward(self, x, momentum=False):
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
        # if self.with_avg_pool:
        #     assert x.dim() == 4, \
        #         "Tensor must has 4 dims, got: {}".format(x.dim())
        # x = x[0:x.shape[0]//2]

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
        # torch.max 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
        # dep_preds = dep_preds.max(dim=1)[0].unsqueeze(1)
        labels = labels.squeeze(1)
        # losses['loss_l1_dep'] = self.l1_loss(dep_preds, labels) * self.l1_loss_weight

        if self.label_norm_cfg is not None:
            dep_predictions = self.depth_unnormalize(dep_preds,
                                                 self.label_norm_cfg['mean'],
                                                 self.label_norm_cfg['std'])

            targets = self.depth_unnormalize(labels,
                                          self.label_norm_cfg['mean'],
                                          self.label_norm_cfg['std'])

        # depth_target = targets[0].to(torch.int32).cpu().numpy().astype(np.uint8)
        # ori_img2_save = mmcv.imwrite(depth_target.transpose((1, 2, 0)),
        #                              "newback/imgs_segs/imgs_head/targets_" + img_metas[0]['ann_info']['seg_map'])

        # depth_prediction = dep_predictions[0].to(torch.int32).cpu().numpy().astype(np.uint8)
        #
        # ori_img2_save = mmcv.imwrite(depth_prediction.transpose((1, 2, 0)),
        #                              "newback/imgs_segs/imgs_head/targets_" + img_metas[0]['ann_info']['seg_map'])

        # if self.edge_loss is not None:
        #     losses['loss_edge'] = self.edge_loss(dep_preds, labels) * (1 - self.edge_weight)

        if self.ssim_loss is not None:
            # 窗口设置为3*3
            losses['loss_ssim_dep'] = self.ssim_loss(dep_predictions, targets) * self.ssim_weght

        return losses

    def dgr_forward(self, x, momentum=False):
        # if self.with_avg_pool:
        #     assert x.dim() == 4, \
        #         "Tensor must has 4 dims, got: {}".format(x.dim())
        # x = x[x.shape[0]//2:]
        out_conv_trans1 = None
        if not momentum:
            out_conv_trans1 = self.dgr_convs(x)
            out_conv_trans1 = self.dgr_convs_bn(out_conv_trans1)

        output = self.convs(x)
        output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        img_head_out_for_moco_64d = self.dgr_linear64d(output_flatten)

        return out_conv_trans1, img_head_out_for_moco_64d  # [img_prediction] if self.img_norm_cfg is not None else [output]

    def dgr_loss(self, img_preds, targets, img_metas=None):
        losses = dict()
        # losses['loss_l1_img'] = self.l1_loss(img_preds, targets) * self.l1_loss_weight

        # img_preds = img_preds[0]
        if self.img_norm_cfg is not None:
            img_predictions = self.im_unnormalize(img_preds,
                                                  self.img_norm_cfg['mean'],
                                                  self.img_norm_cfg['std'],
                                                  self.img_norm_cfg['to_rgb'])

            img_targets = self.im_unnormalize(targets,
                                              self.img_norm_cfg['mean'],
                                              self.img_norm_cfg['std'],
                                              self.img_norm_cfg['to_rgb'])


            img_target = img_targets[0].to(torch.int32).cpu().numpy()
            a = torch.from_numpy(np.array([img_target[2], img_target[1], img_target[0]]))
            ori_img2_save = mmcv.imwrite(file_path="newback/imgs_segs/imgs_head/0311_targets_" + img_metas[0]['ori_filename'],
                                         img=a.to(torch.int32).cpu().numpy().transpose((1, 2, 0)))

            img_prediction = img_predictions[0].to(torch.int32).cpu().numpy()
            a = torch.from_numpy(np.array([img_prediction[2], img_prediction[1], img_prediction[0]]))
            ori_img2_save = mmcv.imwrite(file_path="newback/imgs_segs/imgs_head/0311_img_prediction_" + img_metas[0]['ori_filename'],
                                         img=a.to(torch.int32).cpu().numpy().transpose((1, 2, 0)))

        if self.ssim_loss is not None:
            losses['loss_ssim_img'] = self.ssim_loss(img_predictions, img_targets) * self.ssim_weght

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


