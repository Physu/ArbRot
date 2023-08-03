import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class Cifar100Dataset(CustomDataset):
    """Special SUNRGBD dataset.

    Args:
        split (str): Split txt file for SUNRGBD.
    """

    # cifar100
    CLASSES = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
    'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman',
    'worm')  # ('background', 'airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0]]
    # 注意这里PALETTE的数值，对应的BGR模式，如果是直接读取图片信息，三维信息对应的是RGB模式

    def __init__(self, split, **kwargs):
        super(Cifar100Dataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)


        index = [0,60,120,180,240]
        i = 0
        palette = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    i = i+1
                    pale = [index[i], [index[j], index[k]]]
                    palette.append(pale)
                    if i >100:
                        break

        assert osp.exists(self.img_dir) and self.split is not None
