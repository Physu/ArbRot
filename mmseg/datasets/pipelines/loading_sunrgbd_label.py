import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
import cv2
import imageio
import torchvision.transforms.functional as F

@PIPELINES.register_module()
class LoadSUNRGBDLabel(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow',
                 keep_origin=False,
                 test_mode=False):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.keep_origin = keep_origin
        self.test_mode = test_mode

    def read_annotations(self, label_path):
        item_list = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                item_list.append(str.split(line.rstrip('\n\r')))

        return item_list

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.test_mode:
            results['label'] = None
        else:
            results['label'] = self.read_annotations(osp.join(results['label_prefix'], results['label_info']['label']))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class NormalizeImgAndHHALabel(object):
    """Normalize the image and label

    Added key is "img_norm_cfg and label_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, img_norm_cfg=None):
        self.img_norm_cfg = img_norm_cfg
        if self.img_norm_cfg is not None:
            self.img_mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
            self.img_std = np.array(img_norm_cfg['std'], dtype=np.float32)
            self.to_rgb = img_norm_cfg['to_rgb']

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img = results['img']
        depth = results['gt_semantic_seg']

        img = F.to_tensor(img.copy()).float()  # 这一步相当于0-255，归一化到0-1.0
        depth = F.to_tensor(depth.astype(np.uint8)).float()

        if self.img_norm_cfg is not None:
            img = F.normalize(img, self.img_mean, self.img_std)  # 这一步相当于0-255，归一化到0-1.0
            depth = F.normalize(depth, self.img_mean, self.img_std)

        results['img'] = img
        results['depth'] = depth

        # ori_img1_save = mmcv.imwrite(img_aug1,
        #                              "newback/imgs_segs/imgs_beforenorm/img1_" + results['img_info']['filename'])
        # # ori_img2_save = mmcv.imwrite(img_aug2,
        # #                              "newback/imgs_segs/imgs_beforenorm/img2_" + results['img_info']['filename'])
        # if len(gt_semantic_seg_aug1.astype(np.uint8).shape) == 2 or gt_semantic_seg_aug1.astype(np.uint8).shape[-1] == 1:
        #     img = cv2.cvtColor(gt_semantic_seg_aug1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        #     ori_depth_save = mmcv.imwrite(img,
        #                                   "newback/imgs_segs/imgs_beforenorm/depth1_" + results['ann_info']['seg_map'])
        # 这个sava_image 保存图片顺序rgb
        # ori_depth_save = save_image(gt_semantic_seg_aug1.astype(np.uint8),
        #                              "newback/imgs_segs/debug_nan/depth1_" + results['ann_info']['seg_map'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

