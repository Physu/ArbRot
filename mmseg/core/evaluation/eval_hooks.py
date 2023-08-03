import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import time


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.iter = 0

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        end = time.time()
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        data_time = time.time() - end
        eval_res.update({'data_time': data_time})
        self.iter = self.iter + 1

        print('Val iter: [{current_iter}]\t'
              'Val Time: {batch_time:.3f}\t\n'
              '***********************************************************************\t\n'
              'img_rot_cls_acc: {img_rot_cls_acc:.4f}\t'
              'img_rot_res_gap: {img_rot_res_gap:.4f}\t'
              'img_rot_gap: {img_rot_gap:.4f}\t'
              'img_loc_acc: {img_loc_acc:.4f}\t\n'
              'dep_rot_cls_acc: {depth_rot_cls_acc:.4f}\t'
              'dep_rot_res_gap: {depth_rot_res_gap:.4f}\t'
              'dep_rot_gap: {depth_rot_gap:.4f}\t'
              'dep_loc_acc: {depth_loc_acc:.4f}\t\n'
              'all_rot_cls_acc: {rot_cls_acc:.4f}\t'
              'all_rot_res_gap: {rot_res_gap:.4f}\t'
              'all_rot_gap: {rot_gap:.4f}\t'
              'all_loc_acc: {loc_acc:.4f}\t\n'
              '***********************************************************************\t\n'
              .format(current_iter=self.iter, batch_time=eval_res['data_time'],
                      img_rot_cls_acc=eval_res['img_rot_cls_acc'], img_rot_res_gap=eval_res['img_rot_res_gap'],
                      img_rot_gap=eval_res['img_rot_gap'], img_loc_acc=eval_res['img_loc_acc'],
                      depth_rot_cls_acc=eval_res['depth_rot_cls_acc'],
                      depth_rot_res_gap=eval_res['depth_rot_res_gap'],
                      depth_rot_gap=eval_res['depth_rot_gap'], depth_loc_acc=eval_res['depth_loc_acc'],
                      rot_cls_acc=eval_res['rot_cls_acc'], rot_res_gap=eval_res['rot_res_gap'],
                      rot_gap=eval_res['rot_gap'], loc_acc=eval_res['loc_acc']
                      ))

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        return eval_res

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters
        if current == runner.max_iters-1:  # 训练结束后，evaluation一次
            return True

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.iter = 0

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        end = time.time()
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        data_time = time.time()-end
        eval_res.update({'data_time': data_time})
        self.iter = self.iter + 1

        print('Val iter: [{current_iter}]\t'
              'Val Time: {batch_time:.3f}\t\n'
              '***********************************************************************\t\n'
              'img_rot_cls_acc: {img_rot_cls_acc:.4f}\t'
              'img_rot_res_gap: {img_rot_res_gap:.4f}\t'
              'img_rot_gap: {img_rot_gap:.4f}\t'
              'img_loc_acc: {img_loc_acc:.4f}\t\n'
              'dep_rot_cls_acc: {depth_rot_cls_acc:.4f}\t'
              'dep_rot_res_gap: {depth_rot_res_gap:.4f}\t'
              'dep_rot_gap: {depth_rot_gap:.4f}\t'
              'dep_loc_acc: {depth_loc_acc:.4f}\t\n'
              'all_rot_cls_acc: {rot_cls_acc:.4f}\t'
              'all_rot_res_gap: {rot_res_gap:.4f}\t'
              'all_rot_gap: {rot_gap:.4f}\t'
              'all_loc_acc: {loc_acc:.4f}\t\n'
              '***********************************************************************\t\n'
              .format(current_iter=self.iter, batch_time=eval_res['data_time'],
                      img_rot_cls_acc=eval_res['img_rot_cls_acc'], img_rot_res_gap=eval_res['img_rot_res_gap'],
                      img_rot_gap=eval_res['img_rot_gap'], img_loc_acc=eval_res['img_loc_acc'],
                      depth_rot_cls_acc=eval_res['depth_rot_cls_acc'], depth_rot_res_gap=eval_res['depth_rot_res_gap'],
                      depth_rot_gap=eval_res['depth_rot_gap'], depth_loc_acc=eval_res['depth_loc_acc'],
                      rot_cls_acc=eval_res['rot_cls_acc'], rot_res_gap=eval_res['rot_res_gap'],
                      rot_gap=eval_res['rot_gap'], loc_acc=eval_res['loc_acc']
                      ))

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        return eval_res
