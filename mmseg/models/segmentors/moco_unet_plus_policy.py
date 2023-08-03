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
import os
import pickle
import torch.nn.functional as F
from scipy.special import softmax
from .moco_unet_plus import MoCoUnetPlus
from mmseg.core.utils import SGDPolicy
import itertools
from mmseg.models.segmentors.contra_unet_plus import ContraUnetPlus

layer_dict = {
    'ResNet101': [3, 4, 23, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet34': [3, 4, 6, 3],
    'ResNet18': [2, 2, 2, 2]
    }
@SEGMENTORS.register_module()
class MoCoUnetPlusPolicy(ContraUnetPlus):
    """MOCO.
    1.参考/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/imgs_segs/imgs_head/img_prediction_005265.jpg 得到的图片，效果很不好
    决定先去掉moco，单独训练 rgb generate depth 分支，看看效果如何

    SUNRGBDDoubleHeadUnilateral-> double_generation_head
                                  loc head
                                  rot head

    2. 相对于sunrgbd_doublehead_unilateral.py，将moco重新加回来，完成训练
    3. 相对于moco.py，本版本多了unet来做深度估计
    4. 另外就是再config当中，返回来的是resnet各阶段的输出参数
    5.0321 修改之前的归一化数据，
    6.0405 注意本版本，将之前耦合的代理任务全部解耦
    7.0407 在上一版本基础上，融合rgb和hha的vector，共同构成128D vectors， 并且加入了float weight这个设定
    8.without float weight 这个设定
    9.0801此版本，将moco也调整为可以删除
    10.0903此版本，在0801基础上，rgd和dgr两个loss的计算改为恢复原图，而不是增广后的图片

    11. MoCoUnetPlus 这个版本是为了实现 Adaptice Share, 注意这个需要重新修改Loc head部分，location
        注意修改了，利用centernet来预测location，loc 和 rotation 共用一个loc_neck


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
                 moco_head=None,
                 simsiam_head=None,
                 simclr_head=None,
                 byol_head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 layers='ResNet50',
                 skip_layer=0,
                 init_neg_logits=-10,
                 init_temperature=5.0,
                 temperature_decay=0.965,
                 curriculum_speed=3,
                 hard_sampling=False,
                 init_method='equal',
                 rot_head=None,
                 loc_neck=None,
                 loc_bbox_head=None,
                 generation_neck=None,
                 depth_generate_rgb_head=None,
                 rgb_generate_depth_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 work_dir=None,
                 **kwargs):
        super(MoCoUnetPlusPolicy, self).__init__(
            backbone=backbone,
            neck=neck,
            moco_head=moco_head,
            simsiam_head=simsiam_head,
            simclr_head=simclr_head,
            byol_head=byol_head,
            pretrained=pretrained,
            queue_len=queue_len,
            feat_dim=feat_dim,
            momentum=momentum,
            rot_head=rot_head,
            loc_neck=loc_neck,
            loc_bbox_head=loc_bbox_head,
            generation_neck=generation_neck,
            depth_generate_rgb_head=depth_generate_rgb_head,
            rgb_generate_depth_head=rgb_generate_depth_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg
        )
        # borrow from adashre
        self.init_neg_logits = init_neg_logits
        self.temp = init_temperature
        self._tem_decay = temperature_decay
        self.layers = layer_dict[layers]
        self.skip_layer = skip_layer
        self.policys = []
        self.task_category = []
        if (moco_head or simclr_head or simsiam_head or byol_head):
            self.task_category.append('contra')
        if rot_head is not None:
            self.task_category.append('rot')
        if loc_bbox_head is not None:
            self.task_category.append('loc')
        if rgb_generate_depth_head is not None:
            self.task_category.append('rgd')
        if depth_generate_rgb_head is not None:
            self.task_category.append('dgr')
        for t_id in range(len(self.task_category)):
            self.policys.append(None)
        self.init_method = init_method
        self.reset_logits()  # 定义task logits, 这个就是关键的地方，对最后的权重更新有影响
        self.curriculum_speed = curriculum_speed
        self.hard_sampling = hard_sampling

        self.flag == 'update_w'  # flag 初始化
        # self.optimizer1 = SGDPolicy(itertools.chain(self.backbone_q.parameters(),
        #                                             self.neck_q.parameters(),
        #                                             self.contrastive_head.parameters()), lr=0.001)

        self.first_time_train = True
        self.first_time_val = False

        self.policys_save_dir = '/data1/lhy/InfiRot/mmsegmentation/' + work_dir

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
        if kwargs is not None:  # 用于后面policy的学习
            self.epoch = kwargs['epoch']
            self.num_train_layers = kwargs['num_train_layers']

        losses = self(**data_batch)  # 这里调用了sunrgbd_moco
        loss, log_vars = self._parse_losses(losses)
        losses['rot.loss'] = losses['rot.rot_class_loss'] + losses['rot.rot_res_loss']
        losses.pop('rot.rot_class_loss')
        losses.pop('rot.rot_res_loss')
        losses['loc.loss'] = losses['loc.loss_center_heatmap'] + losses['loc.loss_wh'] + losses['loc.loss_offset']
        losses.pop('loc.loss_center_heatmap')
        losses.pop('loc.loss_wh')
        losses.pop('loc.loss_offset')

        # self.optimizer1.zero_grad()
        grad_dict = dict()
        if self.moco_head is not None:
            losses['loss_img_contrastive'].backward(retain_graph=True)
            contra_grad_dict = dict()
            for name, p in self.backbone_q.named_parameters():
                contra_grad_dict.update({'module.backbone_q.'+name: p.grad.clone().detach()})
            grad_dict.update({'contra_grad': contra_grad_dict})
        if self.simclr_head is not None:
            losses['loss_contrastive'].backward(retain_graph=True)
            contra_grad_dict = dict()
            for name, p in self.backbone_q.named_parameters():
                contra_grad_dict.update({'module.backbone_q.' + name: p.grad.clone().detach()})
            grad_dict.update({'contra_grad': contra_grad_dict})
        if self.simsiam_head is not None:
            losses['loss_simsiam'].backward(retain_graph=True)
            contra_grad_dict = dict()
            for name, p in self.backbone_q.named_parameters():
                contra_grad_dict.update({'module.backbone_q.' + name: p.grad.clone().detach()})
            grad_dict.update({'contra_grad': contra_grad_dict})
        if self.byol_head is not None:
            losses['loss_byol'].backward(retain_graph=True)
            contra_grad_dict = dict()
            for name, p in self.backbone_q.named_parameters():
                contra_grad_dict.update({'module.backbone_q.' + name: p.grad.clone().detach()})
            grad_dict.update({'contra_grad': contra_grad_dict})
        ############################################################
        losses['loc.loss'].backward(retain_graph=True)
        loc_grad_dict = dict()
        for name, p in self.backbone_q.named_parameters():
            loc_grad_dict.update({'module.backbone_q.'+name: p.grad.clone().detach()})
        grad_dict.update({'loc_grad': loc_grad_dict})
        ############################################################
        losses['rot.loss'].backward(retain_graph=True)
        rot_grad_dict = dict()
        for name, p in self.backbone_q.named_parameters():
            rot_grad_dict.update({'module.backbone_q.'+name: p.grad.clone().detach()})
        grad_dict.update({'rot_grad': rot_grad_dict})
        ##########################################################
        losses['rgd.loss_rgd_total'].backward(retain_graph=True)
        rgd_grad_dict = dict()
        for name, p in self.backbone_q.named_parameters():
            rgd_grad_dict.update({'module.backbone_q.'+name: p.grad.clone().detach()})
        grad_dict.update({'rgd_grad': rgd_grad_dict})
        ##########################################################
        losses['dgr.loss_dgr_total'].backward(retain_graph=True)
        dgr_grad_dict = dict()
        for name, p in self.backbone_q.named_parameters():
            dgr_grad_dict.update({'module.backbone_q.'+name: p.grad.clone().detach()})
        grad_dict.update({'dgr_grad': dgr_grad_dict})

        for name, _ in self.backbone_q.named_parameters():
            name = 'module.backbone_q.' + name
            grad_dict['dgr_grad'][name] = grad_dict['dgr_grad'][name] - grad_dict['rgd_grad'][name]
            grad_dict['rgd_grad'][name] = grad_dict['rgd_grad'][name] - grad_dict['rot_grad'][name]
            grad_dict['rot_grad'][name] = grad_dict['rot_grad'][name] - grad_dict['loc_grad'][name]
            grad_dict['loc_grad'][name] = grad_dict['loc_grad'][name] - grad_dict['contra_grad'][name]

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            policys=self.policys,
            grad_dict=grad_dict
        )
        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        if kwargs is not None:  # 用于后面policy的学习
            self.epoch = kwargs['epoch']
            self.num_train_layers = kwargs['num_train_layers']

        losses = self.forward_val(**data_batch)  # 这里调用了sunrgbd_moco
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            policys=self.policys)

        return outputs

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
                      paste_location2=None,
                      ori_img=None,
                      ori_gt_semantic_seg=None
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
        # 这一步，将int64转换为int32，统一尺度
        # 注意这里相当于按图片拼接了

        img_q_and_depth = torch.cat((img[:, 0, ...].contiguous(), gt_semantic_seg_aug1), dim=0)
        img_k_and_depth = torch.cat((img[:, 1, ...].contiguous(), gt_semantic_seg_aug2), dim=0)
        losses = dict()
        split_index = gt_semantic_seg_aug1.shape[0]

        if self.moco_head is not None:
            backbone_q = self.backbone_q(img_q_and_depth)
            # for the NECK operation
            q = self.neck_q(backbone_q)

            # backbone = backbone[0]  # tuple 转成 tensor
            q = nn.functional.normalize(q[0], dim=1)

            # compute key features, use MultiGPU Train
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder_marc()  # update the key encoder
                # 其实这里完全不必管shuffle这个问
                img_k_and_depth, idx_unshuffle = self._batch_shuffle_ddp(img_k_and_depth)  # 双#注释，为了计算计算量，需要恢复
                backbone_k = self.backbone_k(img_k_and_depth)  # momentum update part
                k = self.neck_k(backbone_k)
                k = nn.functional.normalize(k[0], dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)  #
            '''
            注意这里完成了64d->128d 的拼接
            '''
            q = torch.cat((q[0:split_index], q[split_index:]), dim=1)
            k = torch.cat((k[0:split_index], k[split_index:]), dim=1)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            loss_contrastive = self.moco_head(l_pos.cuda(), l_neg.cuda(), flag="img")  # contrastive loss
            losses.update(loss_contrastive)

            self._dequeue_and_enqueue(k)  #

        if self.simsiam_head is not None:
            # backbone_q = self.encoder_q[0](img_q_and_depth)
            backbone_q = self.backbone_q(img_q_and_depth)
            backbone_q2 = self.backbone_q(img_k_and_depth)
            # for the NECK operation
            z1 = self.neck_q(backbone_q)[0]
            z2 = self.neck_q(backbone_q2)[0]
            '''
            注意这里完成了64d->128d 的拼接
            '''
            # loss_contrastive = 0.5 * (self.contrastive_head(z1, z2)['loss'] + self.contrastive_head(z2, z1)['loss'])
            loss_simsiam = self.simsiam_head(z1, z2)['loss']
            losses.update({'loss_simsiam': loss_simsiam})
        if self.simclr_head is not None:
            img = img.reshape(
                (img.size(0) * 2, img.size(2), img.size(3), img.size(4)))
            losses = dict()

            backbone_q = self.backbone_q(img)
            z = self.neck_q(backbone_q)[0]

            z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
            # 为了计算计算量 z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
            assert z.size(0) % 2 == 0
            N = z.size(0) // 2
            s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
            mask, pos_ind, neg_mask = self._create_buffer(N)
            # remove diagonal, (2N)x(2N-1)
            s = torch.masked_select(s.cuda(), mask == 1).reshape(s.size(0), -1)  # s->s.cuda for calculation
            positive = s[pos_ind].unsqueeze(1)  # (2N)x1
            # select negative, (2N)x(2N-2)
            negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
            loss_simclr = self.simclr_head(positive, negative)
            losses.update(loss_simclr)
        #
        if self.byol_head is not None:
            # backbone_q = self.encoder_q[0](img_q_and_depth)
            backbone_q = self.backbone_q(img_q_and_depth)
            backbone_q2 = self.backbone_q(img_k_and_depth)
            # for the NECK operation
            proj_online_v1 = self.neck_q(backbone_q)[0]
            proj_online_v2 = self.neck_q(backbone_q2)[0]
            # backbone = backbone[0]  # tuple 转成 tensor
            # proj_online_v1 = nn.functional.normalize(q[0], dim=1)

            # compute key features, use MultiGPU Train
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder_marc()  # update the key encoder
                backbone_k1 = self.backbone_k(img_q_and_depth)
                backbone_k2 = self.backbone_k(img_k_and_depth)  # momentum update part
                proj_target_v1 = self.neck_k(backbone_k1)[0]
                proj_target_v2 = self.neck_k(backbone_k2)[0]

            '''
            注意这里完成了64d->128d 的拼接
            '''
            loss_byol = 2. * (
                    self.byol_head(proj_online_v1, proj_target_v2)['loss'] +
                    self.byol_head(proj_online_v2, proj_target_v1)['loss'])
            losses.update({'loss_byol': loss_byol})

        if self.loc_neck is not None and (self.loc_head is not None or self.rot_head is not None):
            loc_neck_x = self.loc_neck(backbone_q)
            if self.rot_head is not None:
                loss_rot = self._rot_head_forward_train(loc_neck_x[3], sunrgbd_rotation1.repeat(2), pre_str='rot')
                losses.update(loss_rot)

            if self.loc_head is not None:
                loss_loc = self._loc_head_forward_train(loc_neck_x[3],
                                                        torch.cat((paste_location1, paste_location1), dim=0),
                                                        pre_str='loc', img_metas=img_metas)
                losses.update(loss_loc)
        elif self.loc_neck is None:
            if self.rot_head is not None:
                loss_rot = self._rot_head_forward_train(backbone_q, sunrgbd_rotation1.repeat(2), pre_str='rot')
                losses.update(loss_rot)

            if self.loc_head is not None:
                loss_loc = self._loc_head_forward_train(backbone_q,
                                                        torch.cat((paste_location1, paste_location1), dim=0),
                                                        pre_str='loc', img_metas=img_metas)
                losses.update(loss_loc)

        if self.generation_neck is not None and (
                self.rgb_generate_depth_head is not None or self.depth_generate_rgb_head is not None):
            unet_output = self.generation_neck(backbone_q)
            if self.rgb_generate_depth_head is not None:
                loss_rgd = self._rgd_head_forward_train(unet_output[3][0:split_index], ori_gt_semantic_seg, img_metas,
                                                        pre_str='rgd')
                losses.update(loss_rgd)
            if self.depth_generate_rgb_head is not None:
                loss_dgr = self._dgr_head_forward_train(unet_output[3][split_index:], ori_img, img_metas, pre_str='dgr')
                losses.update((loss_dgr))

        # self.iters = self.iters + 1
        # print(f"self.iters:{self.iters}")
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward_val(self, img, mode='val', **kwargs):
        if mode == 'val':
            # num_train_layers = kwargs['num_train_layers']
            cuda_device = img.get_device()

            # 写成这样，是为了即使 num_train_layer 超过了总的layer层数，也可以继续接着训练，因为温度是逐渐降低的
            num_train_layers = min(sum(self.layers) - self.skip_layer, self.num_train_layers)

            self.policys = self.train_sample_policy(self.temp, hard_sampling=False)
            for task in range(len(self.task_category)):
                self.policys[task] = self.policys[task].to(cuda_device)

            skip_layer = sum(self.layers) - num_train_layers  # 对于resnet50而言 16-1=15
            padding = torch.ones(skip_layer, 2).to(cuda_device)
            padding[:, 1] = 0  # 构建一个二维数组，binary 0，1二值分布
            padding_policys = []

            for t_id in range(len(self.task_category)):  # 对每个任务，都各自进行一次forward()
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)  # 前面构建padding，少了，这里加上，还是对应16层的binary flag
                padding_policys.append(padding_policy)
                setattr(self, '%s_Policy' % (self.task_category[t_id]), padding_policy)

            # kwargs.update({'num_train_layers': num_train_layers})
            # kwargs.update({'is_policy': True})
            # kwargs.update({'policys': padding_policys})

            self.policys = padding_policys

            if self.first_time_val:
                # 只有第一次执行forward的时候，执行下面指令
                self.first_time_train = True
                self.first_time_val = False
                self.fix_weight()
                self.free_alpha()

                self.decay_temperature()  #

                dists = self.get_policy_prob()
                # print(np.concatenate(np.array(dists)))
                self.save_policy()

            return self.forward_train(img, **kwargs)

        else:
            raise Exception("No such mode: {}".format(mode))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            # 因为是curriculum train，下面就是具体的实现方法
            cuda_device = img.get_device()

            num_train_layers = min(sum(self.layers) - self.skip_layer, self.num_train_layers)

            self.policys = self.train_sample_policy(self.temp, hard_sampling=False)  # 这一步比较关键
            for task in range(len(self.task_category)):
                self.policys[task] = self.policys[task].to(cuda_device)

            skip_layer = sum(self.layers) - num_train_layers  # 对于resnet50而言 16-1=15
            padding = torch.ones(skip_layer, 2).to(cuda_device)
            padding[:, 1] = 0  # 构建一个二维数组，binary 0，1二值分布
            padding_policys = []

            for t_id in range(len(self.task_category)):  # 对每个任务，都各自进行一次forward()
                padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()), dim=0)  # 前面构建padding，少了，这里加上，还是对应16层的binary flag
                padding_policys.append(padding_policy)
                setattr(self, '%s_Policy' % (self.task_category[t_id]), padding_policy)

            # kwargs.update({'num_train_layers': num_train_layers})
            # kwargs.update({'is_policy': True})
            # kwargs.update({'policys': padding_policys})
            self.policys = padding_policys
            if self.first_time_train:

                self.first_time_train = False
                self.first_time_val = True
                self.free_weight()
                self.fix_alpha()

            return self.forward_train(img, **kwargs)

        elif mode == 'val':
            # 因为是curriculum train，下面就是具体的实现方法
            # cuda_device = img.get_device()
            #
            # num_train_layers = min(sum(self.layers) - self.skip_layer, self.num_train_layers)
            #
            # self.policys = self.test_sample_policy(hard_sampling=False)
            #
            # for task in range(len(self.task_category)):
            #     self.policys[task] = self.policys[task].to(cuda_device)
            #
            # skip_layer = sum(self.layers) - num_train_layers  # 对于resnet50而言 16-1=15
            # padding = torch.ones(skip_layer, 2).to(cuda_device)
            # padding[:, 1] = 0  # 构建一个二维数组，binary 0，1二值分布
            # padding_policys = []
            #
            # for t_id in range(len(self.task_category)):  # 对每个任务，都各自进行一次forward()
            #     padding_policy = torch.cat((padding.float(), self.policys[t_id][-num_train_layers:].float()),
            #                                dim=0)  # 前面构建padding，少了，这里加上，还是对应16层的binary flag
            #     padding_policys.append(padding_policy)
            #     setattr(self, '%s_Policy' % (self.task_category[t_id]), padding_policy)
            #
            # # kwargs.update({'num_train_layers': num_train_layers})
            # # kwargs.update({'is_policy': True})
            # # kwargs.update({'policys': padding_policys})
            # self.policys = padding_policys
            # if self.first_time_val:
            #     self.first_time_train = True
            #     self.first_time_val = False
            #     self.fix_weight()
            #     self.free_alpha()
            return self.forward_val(img, **kwargs)

        elif mode == 'extract':
            pass
        else:
            raise Exception("No such mode: {}".format(mode))

    ########## 后面这些
    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def sample_policy(self, hard_sampling):
        # dist1, dist2 = self.get_policy_prob()
        # print(np.concatenate((dist1, dist2), axis=-1))
        policys = self.networks['mtl-net'].test_sample_policy(hard_sampling)
        for t_id, p in enumerate(policys):
            setattr(self, 'policy%d' % (t_id + 1), p)

    def save_policy(self, label=None):
        policy = {}
        for task in self.task_category:
            tmp = getattr(self, '%s_Policy' % task)
            policy['%s_Policy' % task] = tmp.cpu().data
        # save_filename = 'policy%s.pickle' % str(label)
        save_filename = 'policy.pickle'
        save_path = os.path.join(self.policys_save_dir, save_filename)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)
        for t_id in range(self.num_tasks):
            setattr(self, 'policy%d' % (t_id + 1), policy['task%d_policy' % (t_id+1)])
            print(getattr(self, 'policy%d'
                          % (t_id + 1)))

    def check_exist_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        return os.path.exists(save_path)

    def fix_weight(self):
        for param in self.backbone_q.parameters():
            param.requires_grad = False

        if self.moco_head or self.byol_head or self.simsiam_head or self.simclr_head:
            if self.moco_head is not None:
                for param in self.moco_head.parameters():
                    param.requires_grad = False
            if self.byol_head is not None:
                for param in self.byol_head.parameters():
                    param.requires_grad = False
            if self.simsiam_head is not None:
                for param in self.simsiam_head.parameters():
                    param.requires_grad = False
            if self.simclr_head is not None:
                for param in self.simclr_head.parameters():
                    param.requires_grad = False
        if self.loc_neck is not None:
            for param in self.loc_neck.parameters():
                param.requires_grad = False
        if self.rot_head is not None:
            for param in self.rot_head.parameters():
                param.requires_grad = False
        if self.loc_head is not None:
            for param in self.loc_head.parameters():
                param.requires_grad = False
        if self.rgb_generate_depth_head is not None:
            for param in self.rgb_generate_depth_head.parameters():
                param.requires_grad = False
        if self.depth_generate_rgb_head is not None:
            for param in self.depth_generate_rgb_head.parameters():
                param.requires_grad = False

    def free_weight(self):
        for param in self.backbone_q.parameters():
            param.requires_grad = True

        if self.moco_head or self.byol_head or self.simsiam_head or self.simclr_head:
            if self.moco_head is not None:
                for param in self.moco_head.parameters():
                    param.requires_grad = True
            if self.byol_head is not None:
                for param in self.byol_head.parameters():
                    param.requires_grad = True
            if self.simsiam_head is not None:
                for param in self.simsiam_head.parameters():
                    param.requires_grad = True
            if self.simclr_head is not None:
                for param in self.simclr_head.parameters():
                    param.requires_grad = True
        if self.loc_neck is not None:
            for param in self.loc_neck.parameters():
                param.requires_grad = True
        if self.rot_head is not None:
            for param in self.rot_head.parameters():
                param.requires_grad = True
        if self.loc_head is not None:
            for param in self.loc_head.parameters():
                param.requires_grad = True
        if self.rgb_generate_depth_head is not None:
            for param in self.rgb_generate_depth_head.parameters():
                param.requires_grad = True
        if self.depth_generate_rgb_head is not None:
            for param in self.depth_generate_rgb_head.parameters():
                param.requires_grad = True

    def fix_alpha(self):
        for task in self.task_category:
            logits = getattr(self, '%s_logits' % (task))
            logits.requires_grad = False

    def free_alpha(self):
        for task in self.task_category:
            logits = getattr(self, '%s_logits' % (task))
            logits.requires_grad = True

    def train_sample_policy(self, temperature, hard_sampling):
        policys = []
        for task in self.task_category:  # 注意这里，如果hard_sampling 是False，则可微；为True，不可微，但是会呈现one-hot
            policy = F.gumbel_softmax(getattr(self, '%s_logits' % (task)), temperature, hard=hard_sampling)
            policys.append(policy)
        return policys

    def reset_logits(self):
        num_layers = sum(self.layers)
        for task_name in self.task_category:
            if self.init_method == 'all_chosen':
                assert(self.init_neg_logits is not None)
                task_logits = self.init_neg_logits * torch.ones(num_layers - self.skip_layer, 2)
                task_logits[:, 0] = 0
            elif self.init_method == 'random':
                task_logits = 1e-3 * torch.randn(num_layers-self.skip_layer, 2)
            elif self.init_method == 'equal':
                task_logits = 0.5 * torch.ones(num_layers-self.skip_layer, 2)
            else:
                raise NotImplementedError('Init Method %s is not implemented' % self.init_method)

            self._arch_parameters = []
            self.register_parameter('%s_logits' % (task_name), nn.Parameter(task_logits, requires_grad=False))
            self._arch_parameters.append(getattr(self, '%s_logits' % (task_name)))

    def test_sample_policy(self, hard_sampling):
        self.policys = []
        if hard_sampling:
            for task in self.task_category:
                task_logits = getattr(self, '%s_logits' % (task))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                policy = np.argmax(logits, axis=1)
                policy = np.stack((1 - policy, policy), dim=1)

                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(policy)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(policy))

                self.policys.append(policy)
        else:
            for task in self.task_category:
                task_logits = getattr(self, '%s_logits' % (task))
                cuda_device = task_logits.get_device()
                logits = task_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).to('cuda:%d' % cuda_device)
                else:
                    policy = torch.from_numpy(np.array(single_policys))

                self.policys.append(policy)

        return self.policys

    def get_policy_prob(self):
        distributions = []
        for task in self.task_category:
            policy = getattr(self, '%s_Policy' % task).detach().cpu().numpy()
            distributions.append(policy)

        return distributions


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
