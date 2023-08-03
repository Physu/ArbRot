import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
from mmcv.cnn import ConvModule
import torch
from mmseg.models import builder
import mmcv
import numpy as np


@HEADS.register_module()
class DoubleGenerationHead(nn.Module):
    """Simplest depth prediction head, with only one fc layer.
    从 rgb 来预测 depth，注意这里的loss计算 即rgd
    从 depth 来预测 rgb，注意这里的loss计算 即dgr
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
    ):
        super(DoubleGenerationHead, self).__init__()
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
                act_cfg=None))

        self.convs = nn.Sequential(*convs)

        self.linear64d = nn.Linear(3 * 32 * 32, 64)  # 输入 841维度，输出64维度
        self.convs_trans1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=(4, 4), stride=4)
        self.convs_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=(2, 2), stride=2)

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

    def rgd_forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())

        out_conv_trans1 = self.convs_trans1(x)
        out_conv_trans1 = self.convs_trans2(out_conv_trans1)
        # 这部分需要产生输出的
        output = self.convs(x)
        output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        dep_head_out_for_moco_64d = self.linear64d(output_flatten)

        return out_conv_trans1, dep_head_out_for_moco_64d

    def rgd_loss(self, dep_preds, labels, img_metas=None):

        # labels1 = labels.permute(1, 2, 0)
        losses = dict()
        # torch.max 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
        dep_preds = dep_preds.max(dim=1)[0].unsqueeze(1)
        losses['loss_dep'] = self.l1_loss(dep_preds, labels) * self.l1_loss_weight

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

        if self.edge_loss is not None:
            losses['loss_edge'] = self.edge_loss(dep_preds, labels) * (1 - self.edge_weight)

        return losses

    def dgr_forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())

        out_conv_trans1 = self.convs_trans1(x)
        out_conv_trans1 = self.convs_trans2(out_conv_trans1)
        # output_flatten = out_conv_trans1.view(out_conv_trans1.size(0), -1)  # 将4维压缩到2维

        output = self.convs(x)
        output_flatten = output.view(output.size(0), -1)  # 将4维压缩到2维
        img_head_out_for_moco_64d = self.linear64d(output_flatten)

        return out_conv_trans1, img_head_out_for_moco_64d  # [img_prediction] if self.img_norm_cfg is not None else [output]

    def dgr_loss(self, img_preds, targets, img_metas=None):
        losses = dict()
        losses['loss_img'] = self.l1_loss(img_preds, targets) * self.l1_loss_weight

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


            # img_target = img_targets[0].to(torch.int32).cpu().numpy()
            # a = torch.from_numpy(np.array([img_target[2], img_target[1], img_target[0]]))
            # ori_img2_save = mmcv.imwrite(file_path="newback/imgs_segs/imgs_head/targets_" + img_metas[0]['ori_filename'],
            #                              img=a.to(torch.int32).cpu().numpy().transpose((1, 2, 0)))

            # img_prediction = img_predictions[0].to(torch.int32).cpu().numpy()
            # a = torch.from_numpy(np.array([img_prediction[2], img_prediction[1], img_prediction[0]]))
            # ori_img2_save = mmcv.imwrite(file_path="newback/imgs_segs/imgs_head/img_prediction_" + img_metas[0]['ori_filename'],
            #                              img=a.to(torch.int32).cpu().numpy().transpose((1, 2, 0)))

        if self.ssim_loss is not None:
            losses['loss_ssim'] = self.ssim_loss(img_predictions, img_targets) * self.ssim_weght

        return losses
