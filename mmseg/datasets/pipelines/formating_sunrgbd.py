from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class DefaultFormatBundle_SUNRGBD(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))  # BGR to RGB
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...]), stack=True)  # 此处的astype(np.int64) 去除
        if 'img_aug1' in results:
            img_aug1 = results['img_aug1']
            img_aug2 = results['img_aug2']
            img_aug1 = np.ascontiguousarray(img_aug1.transpose(2, 0, 1))  # BGR to RGB
            img_aug2 = np.ascontiguousarray(img_aug2.transpose(2, 0, 1))  # BGR to RGB
            results['img_aug1'] = DC(to_tensor(img_aug1), stack=True)
            results['img_aug2'] = DC(to_tensor(img_aug2), stack=True)

        if 'img_aug1_normalization' in results:
            img_aug1_normalization = results['img_aug1_normalization']
            img_aug2_normalization = results['img_aug2_normalization']
            img_aug1_normalization = np.ascontiguousarray(img_aug1_normalization.transpose(2, 0, 1))  # BGR to RGB
            img_aug2_normalization = np.ascontiguousarray(img_aug2_normalization.transpose(2, 0, 1))  # BGR to RGB
            results['img_aug1_normalization'] = DC(to_tensor(img_aug1_normalization), stack=True)
            results['img_aug2_normalization'] = DC(to_tensor(img_aug2_normalization), stack=True)

        if 'gt_semantic_seg_aug1' in results:
            # convert to long
            results['gt_semantic_seg_aug1'] = DC(
                to_tensor(results['gt_semantic_seg_aug1'][None, ...]), stack=True)  # 此处的astype(np.int64) 去除
            results['gt_semantic_seg_aug2'] = DC(
                to_tensor(results['gt_semantic_seg_aug2'][None, ...]), stack=True)

        if 'gt_semantic_seg_aug1_normalization' in results:
            # convert to long
            results['gt_semantic_seg_aug1_normalization'] = DC(
                to_tensor(results['gt_semantic_seg_aug1_normalization'][None, ...]), stack=True)  # 此处的astype(np.int64) 去除
            results['gt_semantic_seg_aug2_normalization'] = DC(
                to_tensor(results['gt_semantic_seg_aug2_normalization'][None, ...]), stack=True)
        # 这部分用于图片和深度估计的loss的恢复
        if 'img_aug1_for_loss_normalization' in results:
            img_aug1_for_loss_normalization = results['img_aug1_for_loss_normalization']
            img_aug2_for_loss_normalization = results['img_aug2_for_loss_normalization']
            img_aug1_for_loss_normalization = np.ascontiguousarray(img_aug1_for_loss_normalization.transpose(2, 0, 1))  # BGR to RGB
            img_aug2_for_loss_normalization = np.ascontiguousarray(img_aug2_for_loss_normalization.transpose(2, 0, 1))  # BGR to RGB
            results['img_aug1_for_loss_normalization'] = DC(to_tensor(img_aug1_for_loss_normalization), stack=True)
            results['img_aug2_for_loss_normalization'] = DC(to_tensor(img_aug2_for_loss_normalization), stack=True)

        if 'gt_semantic_seg_aug1_for_loss_normalization' in results:
            # convert to long
            results['gt_semantic_seg_aug1_for_loss_normalization'] = DC(
                to_tensor(results['gt_semantic_seg_aug1_for_loss_normalization'][None, ...]), stack=True)  # 此处的astype(np.int64) 去除
            results['gt_semantic_seg_aug2_for_loss_normalization'] = DC(
                to_tensor(results['gt_semantic_seg_aug2_for_loss_normalization'][None, ...]), stack=True)

        if 'label' in results:
            label = results['label']
            results['label'] = DC(to_tensor(label), stack=True)

        if 'ori_img' in results:
            ori_img = results['ori_img']
            if len(ori_img.shape) < 3:
                img = np.expand_dims(ori_img, -1)
            ori_img = np.ascontiguousarray(ori_img.transpose(2, 0, 1))  # BGR to RGB
            results['ori_img'] = DC(to_tensor(ori_img), stack=True)

        if 'ori_gt_semantic_seg' in results:
            # convert to long
            results['ori_gt_semantic_seg'] = DC(
                to_tensor(results['ori_gt_semantic_seg'][None,
                                                     ...]), stack=True)  # 此处的astype(np.int64) 去除

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class DefaultFormatBundle_SUNRGBDHHA(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img_aug1' in results:
            img_aug1 = results['img_aug1']
            img_aug1 = np.ascontiguousarray(img_aug1)  # 目前看，这个本来就是RGB格式 不需要transpose(2, 0, 1)
            results['img_aug1'] = DC(to_tensor(img_aug1), stack=True)
            # from torchvision.utils import save_image
            # save_image(img_aug1, "newback/imgs_segs/imgs_beforenorm/img_aug1_for_test.jpg")
        if 'img_aug2' in results:
            img_aug2 = results['img_aug2']
            img_aug2 = np.ascontiguousarray(img_aug2)  # 注意这里， 不需要 BGR to RGB
            results['img_aug2'] = DC(to_tensor(img_aug2), stack=True)

        if 'gt_semantic_seg_aug1' in results:
            gt_semantic_seg_aug1 = results['gt_semantic_seg_aug1']
            results['gt_semantic_seg_aug1'] = DC(
                to_tensor(gt_semantic_seg_aug1), stack=True)

        if 'gt_semantic_seg_aug2' in results:
            gt_semantic_seg_aug2 = results['gt_semantic_seg_aug2']
            results['gt_semantic_seg_aug2'] = DC(
                to_tensor(gt_semantic_seg_aug2), stack=True)
        if 'flip' not in results:
            results['flip'] = None

        if 'flip_direction' not in results:
            results['flip_direction'] = None

        if 'img' in results:
            img = results['img']
            img = np.ascontiguousarray(img)  # 目前看，这个本来就是RGB格式 不需要transpose(2, 0, 1)
            results['img'] = DC(to_tensor(img), stack=True)
        if 'depth' in results:
            depth = results['depth']
            results['depth'] = DC(to_tensor(depth), stack=True)

        if 'label' in results:
            label = results['label'][0]
            results['location'] = to_tensor(int(label[1]))
            results['rotation'] = to_tensor(int(label[2]))
            results['loc_coordinate'] = to_tensor(
                np.array([int(label[3]), int(label[4]), int(label[5]), int(label[6])]))

        if 'ori_img' in results:
            ori_img = results['ori_img']
            ori_img = np.ascontiguousarray(ori_img)  # 目前看，这个本来就是RGB格式 不需要transpose(2, 0, 1)
            results['ori_img'] = DC(to_tensor(ori_img), stack=True)

        if 'ori_gt_semantic_seg' in results:
            ori_gt_semantic_seg = results['ori_gt_semantic_seg']
            ori_gt_semantic_seg = np.ascontiguousarray(ori_gt_semantic_seg)
            results['ori_gt_semantic_seg'] = DC(
                to_tensor(ori_gt_semantic_seg), stack=True)

        # if 'paste_location1' in results:
        #     paste_location1 = results['paste_location1']
        #     results['paste_location1'] = to_tensor(paste_location1)
        #
        # if 'paste_location2' in results:
        #     paste_location2 = results['paste_location2']
        #     results['paste_location2'] = to_tensor(paste_location2)
        #
        # if 'sunrgbd_rotation1' in results:
        #     sunrgbd_rotation1 = results['sunrgbd_rotation1']
        #     results['sunrgbd_rotation1'] = to_tensor(sunrgbd_rotation1)
        #
        # if 'sunrgbd_rotation2' in results:
        #     sunrgbd_rotation2 = results['sunrgbd_rotation2']
        #     results['sunrgbd_rotation2'] = to_tensor(sunrgbd_rotation2)


        return results

    def __repr__(self):
        return self.__class__.__name__