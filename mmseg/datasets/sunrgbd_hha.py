import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from .utils import to_numpy
from PIL import Image
import torch
from .pipelines import Compose
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module()
class SUNRGBDHHADataset(CustomDataset):
    """Special SUNRGBD with HHA dataset.

    Args:
        split (str): Split txt file for SUNRGBD.
    """

    # CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    #            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    #            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor')
    CLASSES = ('background', 'bed', 'table', 'sofa', 'chair', 'toilet',
               'desk', 'dresser', 'night_stand', 'bookshelf', 'bathtub')

    # what's this for?
    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    #            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    #            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
    #            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
               [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0]]
    # 注意这里PALETTE的数值，对应的BGR模式，如果是直接读取图片信息，三维信息对应的是RGB模式

    def __init__(self, split, **kwargs):
        super(SUNRGBDHHADataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        # self.pipeline_moco = Compose(pipeline_moco)

    def __getitem__(self, idx):
        # img = self.data_source.get_sample(idx)
        # assert isinstance(img, Image.Image), \
        #     'The output from the data source must be an Image, got: {}. \
        #     Please ensure that the list file does not contain labels.'.format(
        #     type(img))
        # img1 = self.pipeline(img)
        # img2 = self.pipeline(img)
        # if self.prefetch:
        #     img1 = torch.from_numpy(to_numpy(img1))
        #     img2 = torch.from_numpy(to_numpy(img2))
        # img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        # return dict(img=img_cat)

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
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)  # 调用了transforms 里面的相关函数
        img_aug1 = results['img_aug1'].data
        img_aug2 = results['img_aug2'].data
        # img2 = results2['img'].data
        img = torch.cat((img_aug1.unsqueeze(0), img_aug2.unsqueeze(0)), dim=0)
        img = DC(img, stack=True)
        results['img'] = img
        return results
