import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Cifar10Dataset(CustomDataset):
    """Special SUNRGBD dataset.

    Args:
        split (str): Split txt file for SUNRGBD.
    """

    # cifar10
    CLASSES = ('background', 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0]]
    # 注意这里PALETTE的数值，对应的BGR模式，如果是直接读取图片信息，三维信息对应的是RGB模式

    def __init__(self, split, **kwargs):
        super(Cifar10Dataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
