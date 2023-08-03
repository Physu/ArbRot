from mmcv.runner.hooks import HOOKS
import torch

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from mmcv.runner.hooks.optimizer import OptimizerHook


@HOOKS.register_module()
class OptimizerPolicyHook(OptimizerHook):

    def __init__(self, grad_clip=None, named_params_dict=None):
        self.grad_clip = grad_clip
        self.named_params_dict = named_params_dict

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):

        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step(grad_dict=runner.outputs['grad_dict'], named_params_dict=self.named_params_dict,
                              policys=runner.outputs['policys'])

    '''
    # def after_train_iter(self, runner):
    #     # print('#################################################### ooo ###############')
    #     # j = 0
    #     # for param in runner.model.module.backbone_q.parameters():
    #     #     j = j + 1
    #     #     if j == 53:
    #     #         print(param.grad)
    #     runner.optimizer.zero_grad()
    #     runner.outputs['loss'].backward()
    # 
    #     # print('####################################################')
    #     # j = 0
    #     # for param in runner.model.module.backbone_q.parameters():
    #     #     j = j + 1
    #     #     if j == 53:
    #     #         print(param.grad)
    #     # i = 0
    #     # for param in runner.model.module.backbone_q.parameters():
    #     #     i = i + 1
    #     #     if i =
    #     #     print(param.grad)
    #     # runner.outputs['loss'].backward(retain_graph=True)
    #     # runner.outputs['loss'].backward()
    #     #
    #     # # with torch.autograd.set_detect_anomaly(True):
    #     #     # for param in runner.model.parameters():
    #     #     #     param.requires_grad = False
    #     # if 'loss_img_contrastive' in runner.outputs['losses']:
    #     # #     sequence_tensors = []
    #     # #     for param in runner.model.module.backbone_q.parameters():
    #     # #         param.requires_grad = True
    #     # #         sequence_tensors.append(param)
    #     # #     for param in runner.model.module.neck_q.parameters():
    #     # #         param.requires_grad = True
    #     # #         sequence_tensors.append(param)
    #     # #     for param in runner.model.module.contrastive_head.parameters():
    #     # #         param.requires_grad = True
    #     # #         sequence_tensors.append(param)
    #     #
    #     #     runner.optimizer.zero_grad()
    #     #     # torch.autograd.backward(sequence_tensors)
    #     #     runner.outputs['losses']['loss_img_contrastive'].backward(retain_graph=True)
    #     #     # runner.optimizer.step()
    #     #     # contra_grad = runner.optimizer.clone()
    #     #     # for param in runner.model.parameters():
    #     #     #     param.requires_grad = False
    #     # if 'rot.loss' in runner.outputs['losses']:
    #     #     # for param in runner.model.module.backbone_q.parameters():
    #     #     #     param.requires_grad = True
    #     #     #
    #     #     # for param in runner.model.module.loc_neck.parameters():
    #     #     #     param.requires_grad = True
    #     #     # for param in runner.model.module.loc_head.parameters():
    #     #     #     param.requires_grad = True
    #     #
    #     #     runner.optimizer.zero_grad()
    #     #     runner.outputs['losses']['rot.loss'].backward(retain_graph=True)
    #     #     # runner.optimizer.step()
    #     #     # rot_grad = runner.optimizer.clone()
    #     # if 'loc.loss' in runner.outputs['losses']:
    #     #     runner.optimizer.zero_grad()
    #     #     runner.outputs['losses']['loc.loss'].backward(retain_graph=True)
    #     #     # loc_grad = runner.optimizer.clone()
    #     # if 'rgd.loss_rgd_total' in runner.outputs['losses']:
    #     #     runner.optimizer.zero_grad()
    #     #     runner.outputs['losses']['rgd.loss_rgd_total'].backward(retain_graph=True)
    #     #     # rgd_grad = runner.optimizer.clone()
    #     # if 'dgr.loss_dgr_total' in runner.outputs['losses']:
    #     #     runner.optimizer.zero_grad()
    #     #     runner.outputs['losses']['dgr.loss_dgr_total'].backward(retain_graph=True)
    #     #     # dgr_grad = runner.optimizer.clone()
    # 
    #   
    #     j=0
    #     for param in runner.model.parameters():
    #         j=j+1
    #         if j==50:
    #             p=param.grad
    # 
    # 
    #     losses['rot.loss'].backward(retain_graph=True)
    #     j=0
    #     for param in self.backbone_q.parameters():
    #         j=j+1
    #         if j==46:
    #             p=param
    #  
    # 
    #     if self.grad_clip is not None:
    #         grad_norm = self.clip_grads(runner.model.parameters())
    #         if grad_norm is not None:
    #             # Add grad norm to the logger
    #             runner.log_buffer.update({'grad_norm': float(grad_norm)},
    #                                      runner.outputs['num_samples'])
    #     # for name, value in runner.model.named_parameters():
    #     #     print(name)
    #     runner.optimizer.step(grad_dict=runner.outputs['grad_dict'])
    '''
