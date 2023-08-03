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
class MoCoSingleGPU(BaseSegmentor):
    """MOCO.
    1.参考/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/imgs_segs/imgs_head/img_prediction_005265.jpg 得到的图片，效果很不好
    决定先去掉moco，单独训练 rgb generate depth 分支，看看效果如何

    SUNRGBDDoubleHeadUnilateral-> double_generation_head
                                  loc head
                                  rot head

    2. 相对于sunrgbd_doublehead_unilateral.py，将moco重新加回来，完成训练
    0223修改
    3. 相对于moco，就是单卡训练，本版本主要用于data generation（10000数据量）


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
                 contrastive_head=None,
                 head_depth=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 rot_and_loc_head=None,
                 rot_head=None,
                 loc_head=None,
                 double_generation_head=None,
                 img_norm_cfg=None,
                 label_norm_cfg=None,
                 multigpu_train=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(MoCoSingleGPU, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained  # 从这里看的话，应该只是backbone部分的pretrained weight 被加载了

        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        for param in self.encoder_k.parameters():  # 这部分，Key不更新参数
            param.requires_grad = False

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue for rgb branch
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))  # Adds a persistent buffer to the module.
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        if contrastive_head is not None:
            self.contrastive_head = builder_moco.build_head_moco(contrastive_head)
        else:
            self.contrastive_head = None

        if rot_head is not None:
            self.rot_head = builder_moco.build_head_moco(rot_head)
        else:
            self.rot_head = None

        if loc_head is not None:
            self.loc_head = builder_moco.build_head_moco(loc_head)
        else:
            self.loc_head = None

        if double_generation_head is not None:
            self.double_generation_head = builder_moco.build_head_moco(double_generation_head)
        else:
            self.double_generation_head = None

        if rot_and_loc_head is not None:
            self.rot_and_loc_head = builder_moco.build_head_moco(rot_and_loc_head)
        else:
            self.rot_and_loc_head = None

        if multigpu_train is not None:
            self.multigpu_train = multigpu_train

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.label_norm_cfg = label_norm_cfg

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
        pass

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

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

        from mmcv.cnn.utils.weight_init import initialize
        from mmcv.cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:  # 判断是否完成初始化
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
                if hasattr(m, 'init_weights'):  # 这里不探索父类方法吗？
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
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
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
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

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

    def _rot_and_loc_head_forward_train(self, x, rot, loc, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.rot_and_loc_head.forward(x)
        loss = self.rot_and_loc_head.loss(out, rot, loc)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def _rot_head_forward_train(self, x, rot, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.rot_head.forward(x)
        loss = self.rot_head.loss(out, rot)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def _loc_head_forward_train(self, x, loc, pre_str):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        out = self.loc_head.forward(x)
        loss = self.loc_head.loss(out, loc)

        losses.update(add_prefix(loss, pre_str))
        return losses

    def double_generation_head_forward_train(self,
                                             combination_img_and_gt,
                                             img_concate,
                                             gt_semantic_seg_concate,
                                             img_metas,
                                             return_loss=True,
                                             momentum=False):
        '''

        :param combination_img_and_gt: input is composed of img and depth
        :param img_concate:
        :param gt_semantic_seg_concate:
        :param img_metas:
        :param return_loss:
        :param momentum: whether is momentum branch, in this situation, no need to recover the image and depth
        :return:
        '''
        losses = dict()
        total_128d = None
        ptr = combination_img_and_gt.shape[0]
        if not momentum:  # 判断是来自gradient update or momentum update, 下面执行的是 gradient update
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(combination_img_and_gt[0:ptr//4], momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(combination_img_and_gt[ptr//2:ptr-ptr//4], momentum)
            if return_loss and not momentum:
                loss_rgd = self.double_generation_head.rgd_loss(out_rgd, gt_semantic_seg_concate[:ptr//4, 0].unsqueeze(1),
                                                                img_metas)
                losses.update(add_prefix(loss_rgd, "rgd"))

                loss_dgr = self.double_generation_head.dgr_loss(out_dgr, img_concate[0:ptr//4], img_metas)
                losses.update(add_prefix(loss_dgr, "dgr"))
        else:
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(
                combination_img_and_gt[ptr // 4:ptr // 2], momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(combination_img_and_gt[ptr // 2 + ptr // 4: ptr], momentum)
            if return_loss and not momentum:
                loss_rgd = self.double_generation_head.rgd_loss(out_rgd, gt_semantic_seg_concate[ptr // 4:ptr // 2, 0].unsqueeze(1),
                                                                img_metas)
                losses.update(add_prefix(loss_rgd, "rgd"))

                loss_dgr = self.double_generation_head.dgr_loss(out_dgr, img_concate[ptr // 4:ptr // 2, 0], img_metas)
                losses.update(add_prefix(loss_dgr, "dgr"))

        if out_dgr_for_moco_64d is not None and out_rgd_for_moco_64d is not None:
            total_128d = torch.cat((out_rgd_for_moco_64d, out_dgr_for_moco_64d), dim=1)

        return losses, total_128d


    def double_generation_head_forward_train_v2(self,
                                             combination_img_and_gt,
                                             img_concate=None,
                                             gt_semantic_seg_concate=None,
                                             img_metas=None,
                                             return_loss=True,
                                             momentum=False):
        '''

        :param combination_img_and_gt: input is composed of img and depth
        :param img_concate:
        :param gt_semantic_seg_concate:
        :param img_metas:
        :param return_loss:
        :param momentum: whether is momentum branch, in this situation, no need to recover the image and depth
        :return:
        '''
        losses = dict()
        total_128d = None
        # ptr = combination_img_and_gt.shape[0]
        if not momentum:  # 判断是来自gradient update or momentum update, 下面执行的是 gradient update
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(combination_img_and_gt, momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(combination_img_and_gt, momentum)
            if return_loss and not momentum:
                loss_rgd = self.double_generation_head.rgd_loss(out_rgd, gt_semantic_seg_concate[:, 0].unsqueeze(1),
                                                                img_metas)
                losses.update(add_prefix(loss_rgd, "rgd"))

                loss_dgr = self.double_generation_head.dgr_loss(out_dgr, img_concate, img_metas)
                losses.update(add_prefix(loss_dgr, "dgr"))
        else:
            # 用来输出需要的64d的特征向量
            out_rgd, out_rgd_for_moco_64d = self.double_generation_head.rgd_forward(combination_img_and_gt, momentum)
            out_dgr, out_dgr_for_moco_64d = self.double_generation_head.dgr_forward(combination_img_and_gt, momentum)

        if out_dgr_for_moco_64d is not None and out_rgd_for_moco_64d is not None:
            total_128d = torch.cat((out_rgd_for_moco_64d, out_dgr_for_moco_64d), dim=1)

        return losses, total_128d


    def forward_train(self,
                      img,
                      img_aug1,
                      img_aug2,
                      img_aug1_normalization,
                      img_aug2_normalization,
                      img_metas,
                      gt_semantic_seg_aug1,
                      gt_semantic_seg_aug2,
                      gt_semantic_seg_aug1_normalization,
                      gt_semantic_seg_aug2_normalization,
                      sunrgbd_rotation1=None,
                      sunrgbd_rotation2=None,
                      paste_location1=None,
                      paste_location2=None,
                      img_aug1_for_loss_normalization=None,
                      img_aug2_for_loss_normalization=None,
                      gt_semantic_seg_aug1_for_loss_normalization=None,
                      gt_semantic_seg_aug2_for_loss_normalization=None
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
        # assert img.dim() == 5, \
        #     "Input must have 5 dims, got: {}".format(img.dim())
        # # im_q = img[:, 0, ...].contiguous()
        # im_k = img[:, 1, ...].contiguous()

        # img_concate = torch.cat((img[:, 0, ...].contiguous(), img[:, 1, ...].contiguous()), dim=0)  # 将图片q,k拼接到一起
        # single channel to three channels
        gt_semantic_seg_concat_aug1 = torch.cat((gt_semantic_seg_aug1_normalization,
                                                 gt_semantic_seg_aug1_normalization,
                                                 gt_semantic_seg_aug1_normalization), dim=1)
        gt_semantic_seg_concat_aug2 = torch.cat((gt_semantic_seg_aug2_normalization,
                                                 gt_semantic_seg_aug2_normalization,
                                                 gt_semantic_seg_aug2_normalization), dim=1)

        # gt_semantic_seg_concate = torch.cat((gt_semantic_seg_concat_aug1, gt_semantic_seg_concat_aug2), dim=0)
        '''
        Pytorch里的tensor创建时默认是Torch.FloatTensor类型（torch.float32)
        numpy的默认数据类型为float64，如果根据torch.from_numpy创建tensor，如b = torch.from_numpy(a），
        a和返回的b共享一块内存，因此b的类型也是torch.float64,即double类型。
        如此，在后面计算loss的时候，就会因为32和64（double）不匹配产生各种问题
        '''
        # 这一步，将int64转换为int32，统一尺度

        img_q_and_depth = torch.cat((img[:, 0, ...].contiguous(), gt_semantic_seg_concat_aug1), dim=0)
        img_k_and_depth = torch.cat((img[:, 1, ...].contiguous(), gt_semantic_seg_concat_aug2), dim=0)
        # sunrgbd_rotation_concate = torch.cat((sunrgbd_rotation1.to(torch.int64), sunrgbd_rotation2.to(torch.int64)), dim=0)
        # paste_location_concate = torch.cat((paste_location1.to(torch.int64), paste_location2.to(torch.int64)), dim=0)
        # combination_img_and_gt = torch.cat((img_concate, gt_semantic_seg_concate), dim=0)  # img，img_aug, gt, gt_aug 拼接到一起

        q = self.encoder_q(img_q_and_depth)
        q = nn.functional.normalize(q[0], dim=1)
        q = torch.cat((q[0:q.shape[0]//2], q[q.shape[0]//2:]), dim=1)

        losses = dict()
        # compute key features, use MultiGPU Train
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # 其实这里完全不必管shuffle这个问题
            # img_k_and_depth, idx_unshuffle = self._batch_shuffle_ddp(img_k_and_depth)

            k = self.encoder_k(img_k_and_depth)  # momentum update part
            k = nn.functional.normalize(k[0], dim=1)

            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = torch.cat((k[0:k.shape[0] // 2], k[k.shape[0] // 2:]), dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        loss_contrastive = self.contrastive_head(l_pos, l_neg, flag="img")  # contrastive loss
        losses.update(loss_contrastive)

        self._dequeue_and_enqueue(k)

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
    logits 和 targets 是常见的分类网络的输出和标签了，如果分布式训练，则通过 torch.distributed.all_gather() 这个操作，将各个进程的数据都搜集到一块，然后再处理。
    ogits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        logits_gather_list = [torch.zeros_like(logits) for _ in range(ngpus_per_node)]
        torch.distributed.all_gather(logits_gather_list, logits)
        logits = torch.cat(logits_gather_list, dim=0)

        targets_gather_list = [torch.zeros_like(targets) for _ in range(ngpus_per_node)]
        torch.distributed.all_gather(targets_gather_list, targets)
        targets = torch.cat(targets_gather_list, dim=0)

    accuracy, recall, precision, auc = classification_metrics(logits, targets)
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
