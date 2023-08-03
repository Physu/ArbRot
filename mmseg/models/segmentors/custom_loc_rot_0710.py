import torch
import torch.nn as nn

from mmseg.utils import print_log

from mmseg.models import builder_moco, builder
# from mmseg.models.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.builder import SEGMENTORS
from mmseg.ops import resize
from mmseg.core import add_prefix
import warnings
from collections import defaultdict
from mmcv.utils.logging import get_logger, logger_initialized, print_log
from torchvision.utils import save_image
import time
import numpy as np
from collections import OrderedDict
import torch.distributed as dist


@SEGMENTORS.register_module()
class CustomLocRot0710(BaseSegmentor):
    """
    没别的，就是为了预训练模型

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 contrastive_head=None,
                 pretrained=None,

                 rot_head=None,
                 loc_head=None,
                 generation_neck=None,
                 depth_generate_rgb_head=None,
                 rgb_generate_depth_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(CustomLocRot0710, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained  # 从这里看的话，应该只是backbone部分的pretrained weight 被加载了
        self.backbone_q = builder.build_backbone(backbone)
        # self.neck_q = builder_moco.build_neck_moco(neck)

        if loc_head is not None:
            self.loc_head = builder_moco.build_head_moco(loc_head)
        else:
            self.loc_head = None

        if rot_head is not None:
            self.rot_head = builder_moco.build_head_moco(rot_head)
        else:
            self.rot_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.flag = 0
        self.loss_back = None
        self.iters = 0

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from mmcv.cnn import initialize
        from mmcv.cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return
            # 注意这里进行了修改
            for m in self.children():
                if hasattr(m, 'init_weights'):  # 注意这里是有改动的
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        pass

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        pass

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.self.encoder_q[0](img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        pass

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        pass

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, sunrgbd_rotation=None, sunrgbd_location=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        pass

    def forward_dummy(self, img):
        """Dummy forward function."""
        pass

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)  # 这里调用了sunrgbd_moco
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def _rot_head_forward_train(self, x, rot, pre_str, return_loss=True):
        """Run forward function and calculate loss for rotation head in
        training."""
        losses = dict()
        out = self.rot_head.forward(x)

        if return_loss:
            loss = self.rot_head.loss(out, rot)
            losses.update(add_prefix(loss, pre_str))
        else:
            loss = self.rot_head.rot_evaluation(out, rot)
            # losses['rot_cls_pred'] = rot_cls_prd
            # losses['rot_res_pred'] = rot_res_pred
            losses.update(add_prefix(loss, pre_str))
        return losses

    def _loc_head_forward_train(self, x, loc, pre_str, return_loss=True):
        """Run forward function and calculate loss for location head in
        training."""
        losses = dict()
        out = self.loc_head.forward(x)
        if return_loss:
            loss = self.loc_head.loss(out, loc)
            losses.update(add_prefix(loss, pre_str))
        else:
            _, loc_pred_index = torch.max(out[0], 1)
            if loc_pred_index == loc:
                # losses['loc_class_correct'] = True
                losses.update(add_prefix({'loc_class_correct': True}, pre_str))
            else:
                # losses['loc_class_correct'] = False
                losses.update(add_prefix({'loc_class_correct': False}, pre_str))
        return losses

    def _rgd_head_forward_train(self, x, depth, img_metas, pre_str):
        """Run forward function and calculate loss for RGB Generation Depth head in
        training."""
        losses = dict()
        out = self.rgb_generate_depth_head.forward(x)
        loss = self.rgb_generate_depth_head.loss(out, depth, img_metas)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def _dgr_head_forward_train(self, x, rgb, img_metas, pre_str):
        """Run forward function and calculate loss for Depth Generation RGB head in
        training."""
        losses = dict()
        out = self.depth_generate_rgb_head.forward(x)
        loss = self.depth_generate_rgb_head.loss(out, rgb, img_metas)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def forward_train(self,
                      img,
                      depth,
                      img_metas,
                      rotation=None,
                      location=None,
                      loc_coordinate=None,
                      return_loss=True,  # 用于evaluation部分
                      rescale=True,
                      **kwargs
                      ):
        """Forward computation during training.  主要的训练流程都集中在这里

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled. 归一化之后的图片数据
            img_aug1: 数据增广方法1 构成的集合， 注意这里，没有经过normalization 操作
            img_aug2:


        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        '''
        Pytorch里的tensor创建时默认是Torch.FloatTensor类型（torch.float32)
        numpy的默认数据类型为float64，如果根据torch.from_numpy创建tensor，如b = torch.from_numpy(a），
        a和返回的b共享一块内存，因此b的类型也是torch.float64,即double类型。
        如此，在后面计算loss的时候，就会因为32和64（double）不匹配产生各种问题
        '''
        rotation = rotation.squeeze(1)
        location = location.squeeze(1)
        losses = dict()

        # backbone_q = self.encoder_q[0](img_q_and_depth)
        backbone_img = self.backbone_q(img)
        backbone_depth = self.backbone_q(depth)

        if self.rot_head is not None:
            loss_rot_img = self._rot_head_forward_train(backbone_img, rotation, pre_str='img_rot', return_loss=return_loss)
            loss_rot_depth = self._rot_head_forward_train(backbone_depth, rotation, pre_str='depth_rot', return_loss=return_loss)
            losses.update(loss_rot_img)
            losses.update(loss_rot_depth)

        if self.loc_head is not None:
            loss_loc_img = self._loc_head_forward_train(backbone_img, location, pre_str='loc', return_loss=return_loss)
            loss_loc_depth = self._loc_head_forward_train(backbone_depth, location, pre_str='loc', return_loss=return_loss)
            losses.update(loss_loc_img)
            losses.update(loss_loc_depth)

        # self.iters = self.iters + 1
        # print(f"self.iters:{self.iters}")
        return losses

    def forward_test(self,
                     img,
                     depth,
                     img_metas,
                     rotation=None,
                     location=None,
                     loc_coordinate=None,
                     return_loss=False,  # 用于evaluation部分
                     rescale=False,
                     **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        rotation = rotation.squeeze(1)
        location = location.squeeze(1)
        results = dict()

        # backbone_q = self.encoder_q[0](img_q_and_depth)
        backbone_img = self.backbone_q(img)
        backbone_depth = self.backbone_q(depth)

        if self.rot_head is not None:
            loss_rot_img = self._rot_head_forward_train(backbone_img, rotation, pre_str='img_rot', return_loss=return_loss)
            loss_rot_depth = self._rot_head_forward_train(backbone_depth, rotation, pre_str='depth_rot', return_loss=return_loss)
            results.update(loss_rot_img)
            results.update(loss_rot_depth)

        if self.loc_head is not None:
            loss_loc_img = self._loc_head_forward_train(backbone_img, location, pre_str='img_loc', return_loss=return_loss)
            loss_loc_depth = self._loc_head_forward_train(backbone_depth, location, pre_str='depth_loc', return_loss=return_loss)
            results.update(loss_loc_img)
            results.update(loss_loc_depth)

        # self.iters = self.iters + 1
        # print(f"self.iters:{self.iters}")
        return results

    def forward(self, img, mode='train',  return_loss=True, **kwargs):
        if return_loss:  # val 也是这里
            # print(f'mode:{mode}\n, img:{img}\n,**kwargs:{kwargs}')
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)  # 用于valuation

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred