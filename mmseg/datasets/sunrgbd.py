import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SUNRGBDDataset(CustomDataset):
    """Special SUNRGBD dataset.

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
        super(SUNRGBDDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
