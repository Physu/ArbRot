import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
from mmcv.runner import BaseModule
import torch
import math

from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from ..registry import HEADS
from mmseg.models import builder
from ..utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmseg.utils import collect_env, get_root_logger


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


@HEADS.register_module()
class LocHeadPlus(BaseModule):
    """Simplest classifier head, with only one fc layer.
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_circle=dict(with_circle_iou=False, loss_weight=10),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_weight_all=1.,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(LocHeadPlus, self).__init__()
        # borrow code from mmdetection centernet
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.logger = get_root_logger(log_level='INFO')

        self.loss_center_heatmap = builder.build_loss(loss_center_heatmap)
        self.loss_wh = builder.build_loss(loss_wh)
        self.loss_offset = builder.build_loss(loss_offset)
        self.with_circle_iou = loss_circle['with_circle_iou']
        self.loss_circle_weight = loss_circle['loss_weight']
        self.loss_weight_all = loss_weight_all
        self.iter = 0

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    # def forward(self, x):
    #     if self.with_avg_pool:
    #         x = self.avg_pool(x[3])
    #     x = x.view(x.size(0), -1)
    #     cls_score = self.fc_cls(x)
    #     return [cls_score]

    # def loss(self, cls_score, labels):
    #     losses = dict()
    #     assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
    #     '''
    #      The `input` is expected to contain raw, unnormalized scores for each class.
    #     '''
    #     iou_img2 = self.circle_intersection_over_union(loc_pred_x,
    #                                                    loc_pred_y,
    #                                                    loc_pred_radius,
    #                                                    loc_target[0],
    #                                                    loc_target[1],
    #                                                    loc_target[2] * self.img_hw)
    #     losses['loss_loc'] = self.criterion(cls_score[0], labels) * self.loss_total_weight
    #     return losses
    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        # return multi_apply(self.forward_single, feats)
        center_heatmap_pred = self.heatmap_head(feats).sigmoid()
        wh_pred = self.wh_head(feats)
        offset_pred = self.offset_head(feats)
        return [center_heatmap_pred], [wh_pred], [offset_pred]

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred

    def circle_intersection_over_union(self, x1, y1, r1, x2, y2, r2):

        # The format of the circles is (either detection or ground truth) is [xc, yc, r]
        # x1, y1, r1 = circle1
        # x2, y2, r2 = circle2
        d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        d = torch.sqrt(d2)
        t = ((r1 + r2) ** 2 - d2) * (d2 - (r2 - r1) ** 2)
        if t > 0:  # The circles overlap
            intersectArea = r1 ** 2 * torch.acos((r1 ** 2 - r2 ** 2 + d2) / (2 * d * r1)) + r2 ** 2 * torch.acos(
                (r2 ** 2 - r1 ** 2 + d2) / (2 * d * r2)) - 1 / 2 * torch.sqrt(t)
        elif d > r1 + r2:  # The circles are disjoint
            intersectArea = 0
        else:  # One circle is contained entirely within the other
            intersectArea = math.pi * min(r1, r2) ** 2

        circle1Area = math.pi * r1 ** 2
        circle2Area = math.pi * r2 ** 2

        # Divide the intersection by union of bboxA and bboxB: bboxA U bboxB = bboxA +  bboxB - intersectAB
        overlapRatio = intersectArea / (circle1Area + circle2Area - intersectArea)

        # Return the intersection over union value
        return overlapRatio

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['img_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        # if not self.with_circle_iou:
        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        self.iter = self.iter + 1  # 避免训练时不稳定，后边才引入CircleIoU m

        if self.with_circle_iou:  # and self.iter > 500:
            pred_boxes = self.decode_heatmap(center_heatmap_pred,
                                             wh_pred,
                                             offset_pred,
                                             [256, 256],
                                             k=1,
                                             kernel=3)
            sum_IoU = 0.
            for gt_box, pred_box in zip(gt_bboxes, pred_boxes[0]):
                x = (gt_box[0] + gt_box[2]) / 2
                y = (gt_box[1] + gt_box[3]) / 2
                r = (gt_box[2] - gt_box[0]) / 2
                pred_x = (pred_box[0][0] + pred_box[0][2]) / 2
                pred_y = (pred_box[0][1] + pred_box[0][3]) / 2
                pred_r = (pred_box[0][2] - pred_box[0][0]) / 2
                circle_IoU = self.circle_intersection_over_union(x, y, r, pred_x, pred_y, pred_r)
                circle_IoU = torch.clamp(circle_IoU, min=1e-2, max=0.99)
                sum_IoU += 1. - circle_IoU
            print(f'sum_iou:{sum_IoU}, gt_bboxes num:{len(gt_bboxes)}')

            loss_cirIoU = torch.true_divide(sum_IoU * self.loss_circle_weight * self.loss_weight_all,
                              len(gt_bboxes))

            # self.logger.info(
            #     'LocPlus info:\n' + 'the loss_cirIoU exists problem!' + f'circle_IoU:{circle_IoU.item()}' + '\n')
            # return dict(loss_center_heatmap=loss_center_heatmap * self.loss_weight_all,
            #             loss_wh=loss_wh * self.loss_weight_all,
            #             loss_offset=loss_offset * self.loss_weight_all,
            #             loss_cirIoU=torch.tensor(0.).cuda())

            if torch.isnan(loss_cirIoU).any():
                self.logger.info('LocPlus info:\n' + 'the loss_cirIoU exists problem!' + f'circle_IoU:{circle_IoU.item()}' + '\n')
                # print(f'circle_IoU:{circle_IoU}')
                return dict(loss_center_heatmap=loss_center_heatmap * self.loss_weight_all,
                            loss_wh=loss_wh * self.loss_weight_all,
                            loss_offset=loss_offset * self.loss_weight_all,
                            loss_cirIoU=torch.tensor(0.).cuda())
            else:
                return dict(loss_center_heatmap=loss_center_heatmap * self.loss_weight_all,
                            loss_wh=loss_wh * self.loss_weight_all,
                            loss_offset=loss_offset * self.loss_weight_all,
                            loss_cirIoU=torch.true_divide(sum_IoU * self.loss_circle_weight * self.loss_weight_all,
                                                          len(gt_bboxes)))


        return dict(
            loss_center_heatmap=loss_center_heatmap * self.loss_weight_all,
            loss_wh=loss_wh * self.loss_weight_all,
            loss_offset=loss_offset * self.loss_weight_all)

    def get_targets(self, gt_bboxes, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            del gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])

        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])
        center_heatmap_target = center_heatmap_target.float()
        wh_target = wh_target.float()
        offset_target = offset_target.float()
        wh_offset_target_weight = wh_offset_target_weight.float()

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            # gt_label = gt_labels[batch_id]
            ctx = (gt_bbox[0] + gt_bbox[2]) * width_ratio / 2
            cty = (gt_bbox[1] + gt_bbox[3]) * height_ratio / 2
            # gt_centers = torch.cat(([center_x], [center_y]), dim=0)

            # for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ctx.int(), cty.int()  # 获取位置的int 值
            # ctx, cty = ct
            scale_box_h = (gt_bbox[3] - gt_bbox[1]) * height_ratio
            scale_box_w = (gt_bbox[2] - gt_bbox[0]) * width_ratio
            radius = gaussian_radius([scale_box_h, scale_box_w],
                                     min_overlap=0.3)
            radius = max(0, int(radius))
            # ind = gt_label[j]
            center_heatmap_target[batch_id] = gen_gaussian_target(center_heatmap_target[batch_id].float().squeeze(0),
                                                                  [ctx_int, cty_int], radius)

            wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
            wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

            offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
            offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

            wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())

        # preds = self.decode_heatmap(center_heatmap_target, wh_target, offset_target, img_shape,)
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        border_pixs = [img_meta['border'] for img_meta in img_metas]

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            img_metas[0]['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        batch_border = batch_det_bboxes.new_tensor(
            border_pixs)[:, [2, 0, 2, 0]].unsqueeze(1)
        batch_det_bboxes[..., :4] = batch_det_bboxes[..., :4] - batch_border

        if rescale:
            batch_det_bboxes[..., :4] = batch_det_bboxes[..., :4] / batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=1,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape[0:2]

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels
