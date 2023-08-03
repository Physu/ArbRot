# _*_ coding: utf-8 _*_
import scipy.io as sio
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from shutil import copytree, ignore_patterns, copy


train = r'./data/sunrgbd/sunrgbd_trainval/train_data_idx_backup.txt'
with open(train, "w") as f:
    for idx in range(5285):
        idx_str = f'{idx + 5051:06d}'
        f.write(idx_str+"\n")
        print(f'{idx + 5051:06d}')


val = r'./data/sunrgbd/sunrgbd_trainval/val_data_idx_backup.txt'
with open(val, "w") as f:
    for idx in range(5050):
        idx_str = f'{idx + 1:06d}'
        f.write(idx_str+"\n")
        print(f'{idx + 1:06d}')