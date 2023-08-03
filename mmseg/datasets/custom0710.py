import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from .utils import to_numpy
from PIL import Image
import torch
from .pipelines import Compose
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import mmcv
from prettytable import PrettyTable
from mmseg.core import eval_metrics
from collections import OrderedDict
from functools import reduce
import numpy as np
import os


@DATASETS.register_module()
class Custom0710Dataset(CustomDataset):
    """Special SUNRGBD with HHA dataset.

    Args:
        split (str): Split txt file for SUNRGBD.
    """

    # 注意这里PALETTE的数值，对应的BGR模式，如果是直接读取图片信息，三维信息对应的是RGB模式

    def __init__(self, split, ann_suffix='.txt', label_dir=None,  **kwargs):
        self.ann_suffix = ann_suffix
        self.label_dir = label_dir

        super(Custom0710Dataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.jpg', split=split, **kwargs)
        if not (self.label_dir is None or osp.isabs(self.label_dir)):
            self.label_dir = osp.join(self.data_root, self.label_dir)

        assert osp.exists(self.img_dir) and self.split is not None

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        label_info = self.img_infos[idx]['label']
        results = dict(img_info=img_info, ann_info=ann_info, label_info=label_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)  # 调用了transforms 里面的相关函数
        # img = results['img_aug1'].data
        # img = DC(img, stack=True)
        # results['img'] = img
        return results

    def prepare_test_img(self, idx):

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        label_info = self.img_infos[idx]['label']
        results = dict(img_info=img_info, ann_info=ann_info, label_info=label_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                    if ann_dir is not None:
                        ann_name = img_name + self.ann_suffix        # self.read_annotations(img_name)
                        img_info['label'] = dict(label=ann_name)
                    img_infos.append(img_info)
        else:
            # 用于test，目前未实现，如果要用，需要进行修改
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

        results['label_prefix'] = self.label_dir

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset. 这里就获得了所有的evaluation results
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        eval_results = {}

        eval_data_len = len(results)
        img_rot_cls_acc = 0.
        img_rot_res_gap = 0.
        img_rot_gap = 0.
        img_loc_acc = 0.

        depth_rot_cls_acc = 0.
        depth_rot_res_gap = 0.
        depth_rot_gap = 0.
        depth_loc_acc = 0.

        for result in results:
            if result['img_rot.rot_class_correct'] == True:
                img_rot_cls_acc = img_rot_cls_acc + 1
            img_rot_res_gap = img_rot_res_gap + result['img_rot.rotation_res_gap']
            img_rot_gap = img_rot_gap + result['img_rot.rotation_gap']
            if result['img_loc.loc_class_correct'] == True:
                img_loc_acc = img_loc_acc + 1

            # depth part
            if result['depth_rot.rot_class_correct'] == True:
                depth_rot_cls_acc = depth_rot_cls_acc + 1
            depth_rot_res_gap = depth_rot_res_gap + result['depth_rot.rotation_res_gap']
            depth_rot_gap = depth_rot_gap + result['depth_rot.rotation_gap']
            if result['depth_loc.loc_class_correct'] == True:
                depth_loc_acc = depth_loc_acc + 1

        img_rot_cls_acc = img_rot_cls_acc / eval_data_len
        img_rot_res_gap = img_rot_res_gap / eval_data_len
        img_rot_gap = img_rot_gap / eval_data_len
        img_loc_acc = img_loc_acc / eval_data_len

        depth_rot_cls_acc = depth_rot_cls_acc / eval_data_len
        depth_rot_res_gap = depth_rot_res_gap / eval_data_len
        depth_rot_gap = depth_rot_gap / eval_data_len
        depth_loc_acc = depth_loc_acc / eval_data_len

        rot_cls_acc = (img_rot_cls_acc+depth_rot_cls_acc)/2
        rot_res_gap = (img_rot_res_gap+depth_rot_res_gap)/2
        rot_gap = (img_rot_gap+depth_rot_gap)/2
        loc_acc = (img_loc_acc+depth_loc_acc)/2

        eval_results.update({'img_rot_cls_acc': img_rot_cls_acc*100})
        eval_results.update({'img_rot_res_gap': img_rot_res_gap.item()*100})
        eval_results.update({'img_rot_gap': img_rot_gap.item()})
        eval_results.update({'img_loc_acc': img_loc_acc*100})

        eval_results.update({'depth_rot_cls_acc': depth_rot_cls_acc*100})
        eval_results.update({'depth_rot_res_gap': depth_rot_res_gap.item()*100})
        eval_results.update({'depth_rot_gap': depth_rot_gap.item()})
        eval_results.update({'depth_loc_acc': depth_loc_acc*100})

        eval_results.update({'rot_cls_acc': rot_cls_acc*100})
        eval_results.update({'rot_res_gap': rot_res_gap.item()*100})
        eval_results.update({'rot_gap': rot_gap.item()})
        eval_results.update({'loc_acc': loc_acc*100})

        return eval_results


