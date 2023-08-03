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
# torch.set_default_tensor_type(torch.DoubleTensor)


@SEGMENTORS.register_module()
class SUNRGBDMOCOSimpleHeadBilateral(BaseSegmentor):
    """MOCO.
    1.使用了moco，但是为了尽量保全backbone部分的信息，采用可简单的全连接来做最后的分类操作
    不像 sunrgbd_moco.py 里面用复杂的aspp来进行最后的分类和深度恢复操作
    2.和sunrgbd_moco_simplehead.py 相比，正式采用进行mocov2操作，然后进行下游任务

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
                 backbone_depth=None,
                 neck_depth=None,
                 head=None,
                 head_depth=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 label_norm_cfg=None,
                 decode_head=None,  # 添加之前没有的参数
                 auxiliary_head=None,
                 rot_head=None,
                 loc_head=None,
                 dep_head=None,
                 img_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(SUNRGBDMOCOSimpleHeadBilateral, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained  # 从这里看的话，应该只是backbone部分的pretrained weight 被加载了
        # 这里注意，创建了两个twin，用于rgbd 来预测 depth信息
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder_moco.build_neck_moco(neck))  # 注意这里的neck
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder_moco.build_neck_moco(neck))
        # 这里注意，创建了两个twin，用于depth 来预测 rgbd信息
        self.encoder_depth_q = nn.Sequential(
            builder.build_backbone(backbone_depth), builder_moco.build_neck_moco(neck_depth))  # 注意这里的neck
        self.encoder_depth_k = nn.Sequential(
            builder.build_backbone(backbone_depth), builder_moco.build_neck_moco(neck_depth))

        # self.backbone = self.encoder_q[0]
        # self.neck_moco = self.encoder_q[1]
        # if decode_head is not None:
        #     self._init_decode_head(decode_head)  # 这个是deeplabv3部分的代码
        #     self._init_auxiliary_head(auxiliary_head)  # FCN部分的代码

        self.rot_head = rot_head
        self.loc_head = loc_head
        self.dep_head = dep_head
        self.img_head = img_head

        if rot_head is not None:
            self.rot_head = builder_moco.build_head_moco(rot_head)
        if loc_head is not None:
            self.loc_head = builder_moco.build_head_moco(loc_head)
        if dep_head is not None:
            self.dep_head = builder_moco.build_head_moco(dep_head)
        if img_head is not None:
            self.img_head = builder_moco.build_head_moco(img_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.label_norm_cfg = label_norm_cfg

        for param in self.encoder_k.parameters():  # 这部分，Key不更新参数
            param.requires_grad = False
        for param_depth in self.encoder_depth_k.parameters():  # 这部分，Key不更新参数
            param_depth.requires_grad = False
        self.head = builder_moco.build_head_moco(head)  # 创建所需要的head
        self.head_depth = builder_moco.build_head_moco(head_depth)  # 创建所需要的head

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue for rgb branch
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))  # Adds a persistent buffer to the module.
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue for depth branch
        self.register_buffer("queue_depth", torch.randn(feat_dim, queue_len))  # Adds a persistent buffer to the module.
        self.queue_depth = nn.functional.normalize(self.queue_depth, dim=0)
        self.register_buffer("queue_ptr_depth", torch.zeros(1, dtype=torch.long))

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

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
        losses = dict()
        # 主要来自aspphead
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'encode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, sunrgbd_rotation=None, sunrgbd_location=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train_auxiliary(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, sunrgbd_rotation, sunrgbd_location)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train_auxiliary(
                x, img_metas, gt_semantic_seg, self.train_cfg, sunrgbd_rotation, sunrgbd_location)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def init_weights_moco(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')  # pretrained
        self.encoder_q[0].init_weights(pretrained=pretrained)  # 初始化backbone
        self.encoder_q[1].init_weights(init_linear='kaiming')  # 初始化neck
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

        self.encoder_depth_q[0].init_weights(pretrained=pretrained)  # 初始化backbone
        self.encoder_depth_q[1].init_weights(init_linear='kaiming')  # 初始化neck
        for param_depth_q, param_depth_k in zip(self.encoder_depth_q.parameters(),
                                    self.encoder_depth_k.parameters()):
            param_depth_k.data.copy_(param_depth_q.data)

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

            for m in self.children():
                if hasattr(m, 'init_weights'):
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


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _momentum_update_key_encoder_depth(self):
        """Momentum update of the key encoder.动量更新depth branch的权重
        """
        for param_depth_q, param_depth_k in zip(self.encoder_depth_q.parameters(), self.encoder_depth_k.parameters()):
            param_depth_k.data = param_depth_k.data * self.momentum + \
                           param_depth_q.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue_moco(self, keys):
        """Update queue. 这个用于单卡训练"""
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_moco_depth(self, keys):
        """Update queue. 这个用于单卡训练"""
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_depth)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_depth[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr_depth[0] = ptr


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

    def _rot_head_forward_train(self, x, rot, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.rot_head.forward(x)
        loss = self.rot_head.loss(out, rot)

        losses.update(add_prefix(loss, 'rot'+pre_str))
        return losses

    def _loc_head_forward_train(self, x, loc, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.loc_head.forward(x)
        loss = self.loc_head.loss(out, loc)

        losses.update(add_prefix(loss, 'loc'+pre_str))
        return losses

    def _dep_head_forward_train(self, x, gt_semantic_seg, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.dep_head(x)
        loss = self.dep_head.loss(out, gt_semantic_seg)

        losses.update(add_prefix(loss, 'dep'+pre_str))
        return losses

    def _img_head_forward_train(self, x, img, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.img_head(x)
        loss = self.img_head.loss(out, img)

        losses.update(add_prefix(loss, 'img'+pre_str))
        return losses

    def forward_train(self,
                      img,
                      img_aug1,
                      img_aug2,
                      img_metas,
                      gt_semantic_seg_aug1,
                      gt_semantic_seg_aug2,
                      sunrgbd_rotation1=None,
                      sunrgbd_rotation2=None,
                      paste_location1=None,
                      paste_location2=None):
        """Forward computation during training.  主要的训练流程都集中在这里

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.


        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()

        gt_semantic_seg_concat_aug1 = torch.cat((gt_semantic_seg_aug1, gt_semantic_seg_aug1, gt_semantic_seg_aug1), dim=1)
        gt_semantic_seg_concat_aug2 = torch.cat((gt_semantic_seg_aug2, gt_semantic_seg_aug2, gt_semantic_seg_aug2), dim=1)
        '''
        Pytorch里的tensor创建时默认是Torch.FloatTensor类型（torch.float32)
        numpy的默认数据类型为float64，如果根据torch.from_numpy创建tensor，如b = torch.from_numpy(a），
        a和返回的b共享一块内存，因此b的类型也是torch.float64,即double类型。
        如此，在后面计算loss的时候，就会因为32和64（double）不匹配产生各种问题
        '''
        sunrgbd_rotation1 = sunrgbd_rotation1.to(torch.int64)  # 这一步，将int64转换为int32，统一尺度
        sunrgbd_rotation2 = sunrgbd_rotation2.to(torch.int64)
        paste_location1 = paste_location1.to(torch.int64)
        paste_location2 = paste_location2.to(torch.int64)
        # compute query features
        # q = self.encoder_q(im_q)[0]  # queries: NxC
        # 下面的处理是为了将所有的图片都参与训练，先将aug1和aug2都输入backbone得到相关特征
        im_q_backbone = self.encoder_q[0](im_q)
        im_k_backbone = self.encoder_q[0](im_k)
        q = self.encoder_q[1](im_q_backbone)[0]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg, flag="img")  # contrastive loss
        self._dequeue_and_enqueue_moco(k)

        ###########################################################################
        '''
        下面这部分是处理depth的分支
        '''
        depth_q_backbone = self.encoder_depth_q[0](gt_semantic_seg_concat_aug1)
        depth_k_backbone = self.encoder_depth_q[0](gt_semantic_seg_concat_aug2)

        q_depth = self.encoder_depth_q[1](depth_q_backbone)[0]  # neck 部分的代码
        q_depth = nn.functional.normalize(q_depth, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_depth()  # update the key encoder

            k_depth = self.encoder_depth_k(gt_semantic_seg_concat_aug2)[0]  # keys: NxC
            k_depth = nn.functional.normalize(k_depth, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_depth = torch.einsum('nc,nc->n', [q_depth, k_depth]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_depth = torch.einsum('nc,ck->nk', [q_depth, self.queue_depth.clone().detach()])

        losses_depth = self.head(l_pos_depth, l_neg_depth, flag='depth')  # contrastive loss
        self._dequeue_and_enqueue_moco_depth(k_depth)
        losses.update(losses_depth)
        # # 下面是deeplabv3部分
        # losses_deeplabv3 = dict()
        # # ASPPhead
        # loss_decode_semantic_aug1 = self._decode_head_forward_train(im_q_backbone, img_metas,
        #                                               gt_semantic_seg_aug1)
        # loss_decode_semantic_aug2 = self._decode_head_forward_train(im_k_backbone, img_metas,
        #                                               gt_semantic_seg_aug2)
        # losses_deeplabv3.update(add_prefix(loss_decode_semantic_aug1, 'semantic_aug1'))
        # losses_deeplabv3.update(add_prefix(loss_decode_semantic_aug2, 'semantic_aug2'))
        # # FCNhead
        # if self.with_auxiliary_head:
        #     loss_aux_aug1 = self._auxiliary_head_forward_train(
        #         im_q_backbone, img_metas, gt_semantic_seg_aug1, sunrgbd_rotation1, paste_location1)
        #     losses_deeplabv3.update(add_prefix(loss_aux_aug1, 'rot_loc_aug1'))
        #     loss_aux_aug2 = self._auxiliary_head_forward_train(
        #         im_k_backbone, img_metas, gt_semantic_seg_aug2, sunrgbd_rotation2, paste_location2)
        #     losses_deeplabv3.update(add_prefix(loss_aux_aug2, 'rot_loc_aug2'))

        # losses.update(losses_deeplabv3)

        ###############################################################################################
        # 这部分是极简版本的rotation location depth 的预测
        loss_simplified = dict()
        # out_rot_im_q = self.rot_head.forward(im_q_backbone)
        # out_rot_im_k = self.rot_head.forward(im_k_backbone)
        # loss_rot_im_q = self.rot_head.losses(out_rot_im_q,  sunrgbd_rotation1)
        # loss_rot_im_k = self.rot_head.losses(out_rot_im_k, sunrgbd_rotation2)
        if self.rot_head is not None:

            loss_rot_im_q = self._rot_head_forward_train(im_q_backbone[3], sunrgbd_rotation1, "im1")
            loss_rot_im_k = self._rot_head_forward_train(im_k_backbone[3], sunrgbd_rotation2, "im2")
            loss_rot_depth_q = self._rot_head_forward_train(depth_q_backbone[3], sunrgbd_rotation1, "dt1")
            loss_rot_depth_k = self._rot_head_forward_train(depth_k_backbone[3], sunrgbd_rotation2, "dt2")
            losses.update(loss_rot_im_q)
            losses.update(loss_rot_im_k)
            losses.update(loss_rot_depth_q)
            losses.update(loss_rot_depth_k)
        if self.loc_head is not None:
            loss_loc_im_q = self._loc_head_forward_train(im_q_backbone[3], paste_location1, "im1")
            loss_loc_im_k = self._loc_head_forward_train(im_k_backbone[3], paste_location2, "im2")
            loss_loc_depth_q = self._loc_head_forward_train(depth_q_backbone[3], paste_location1, "dt1")
            loss_loc_depth_k = self._loc_head_forward_train(depth_k_backbone[3], paste_location2, "dt2")
            losses.update(loss_loc_im_q)
            losses.update(loss_loc_im_k)
            losses.update(loss_loc_depth_q)
            losses.update(loss_loc_depth_k)
        if self.dep_head is not None:
            loss_dep_im_q = self._dep_head_forward_train(im_q_backbone[3], gt_semantic_seg_aug1, "png")
            loss_dep_im_k = self._dep_head_forward_train(im_k_backbone[3], gt_semantic_seg_aug2, "png")
            loss_dep_depth_q = self._img_head_forward_train(depth_q_backbone[3], img_aug1, "rgb")
            loss_dep_depth_k = self._img_head_forward_train(depth_k_backbone[3], img_aug2, "rgb")

            losses.update(loss_dep_im_q)
            losses.update(loss_dep_im_k)
            # 关于depth的更新

            losses.update(loss_dep_depth_q)
            losses.update(loss_dep_depth_k)
        #######################################################################################################

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.encoder_q[0](img)
        else:
            raise Exception("No such mode: {}".format(mode))

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


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
