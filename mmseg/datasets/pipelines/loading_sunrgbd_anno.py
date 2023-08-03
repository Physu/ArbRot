import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES
import cv2
import imageio

@PIPELINES.register_module()
class LoadSUNRGBDAnnotations(object):
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

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.test_mode:
            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)

            if results.get('seg_prefix', None) is not None:
                filename = osp.join(results['seg_prefix'],
                                    results['ann_info']['seg_map'])
            else:
                filename = results['ann_info']['seg_map']
            img_bytes = self.file_client.get(filename)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze()  # .astype(np.uint8)  # 此处修改，为了读取16位的png图片

            gt_semantic_seg_uint8 = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)  # 此处修改，为了读取16位的png图片 这个相当于将这个数值//256之后的余数

            if not self.keep_origin:
                # change png to depth
                gt_semantic_seg = gt_semantic_seg / 10000.0
            # modify if custom classes
            # if results.get('label_map', None) is not None:
            #     for old_id, new_id in results['label_map'].items():
            #         gt_semantic_seg[gt_semantic_seg == old_id] = new_id
            # # reduce zero_label
            # if self.reduce_zero_label:
            #     # avoid using underflow conversion
            #     gt_semantic_seg[gt_semantic_seg == 0] = 255
            #     gt_semantic_seg = gt_semantic_seg - 1
            #     gt_semantic_seg[gt_semantic_seg == 254] = 255

            # outputImg8U = cv2.convertScaleAbs(gt_semantic_seg, alpha=(255.0 / 65535.0))
            # # seg_crop_save2 = mmcv.imwrite(outputImg8U,
            # #                               "newback/imgs_segs/convert_scale_" + results['img_info']['ann']['seg_map'])  # results['seg_map'])
            # # seg_crop_save2 = mmcv.imwrite(gt_semantic_seg_uint8,
            # #                               "newback/imgs_segs/ori_uint8_" + results['img_info']['ann']['seg_map'])#results['seg_map'])
            # imageio.imwrite("newback/imgs_segs/ori_uint8_" + results['img_info']['ann']['seg_map'], gt_semantic_seg)

            results['gt_semantic_seg'] = gt_semantic_seg
            results['ori_gt_semantic_seg'] = gt_semantic_seg
            results['seg_fields'].append('gt_semantic_seg')
        else:
            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)

            if results.get('seg_prefix', None) is not None:
                filename = osp.join(results['seg_prefix'],
                                    results['ann_info']['seg_map'])
            else:
                filename = results['ann_info']['seg_map']
            img_bytes = self.file_client.get(filename)
            '''
            整型分为有符号整型和无符号整型，其区别在于无符号整型可以存放的正数范围比有符号整型大一倍，因为有符号整型将最高位存储符号，而无符号整型全部存储数字。
            比如16位系统中的一个int能存储的数据范围位-32768-32768，而unsigned能存储的数据范围则是0-65535。
            通俗解释就是 无符号整型中只有正数，他把负数用正数表示。
            '''
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint16)  # 此处修改，为了读取16位的png图片

            if not self.keep_origin:
                # change png to depth
                gt_semantic_seg = gt_semantic_seg/10000.0
            # modify if custom classes
            # if results.get('label_map', None) is not None:
            #     for old_id, new_id in results['label_map'].items():
            #         gt_semantic_seg[gt_semantic_seg == old_id] = new_id
            # # reduce zero_label
            # if self.reduce_zero_label:
            #     # avoid using underflow conversion
            #     gt_semantic_seg[gt_semantic_seg == 0] = 255
            #     gt_semantic_seg = gt_semantic_seg - 1
            #     gt_semantic_seg[gt_semantic_seg == 254] = 255

            # outputImg8U = cv2.convertScaleAbs(gt_semantic_seg, alpha=(255.0 / 65535.0))

            # imageio.imwrite("newback/imgs_segs/gt_loading/imageio_save_" + results['img_info']['ann']['seg_map'], gt_semantic_seg)
            results['gt_semantic_seg'] = gt_semantic_seg
            results['ori_gt_semantic_seg'] = gt_semantic_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
