# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : data_summary.py
# Author: Physu
# Date  : 2022/7/25
# Desc  :
import os

path = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/newback/data/sunrgbd_hha_generate/label'

def read_annotations(label_path):
    item_list = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            item_list.append(str.split(line.rstrip('\n\r')))

    return item_list[0]


rot_class = []
location_class =[]

indicator = 0

file_name_list = os.listdir(path)
for file_name in file_name_list:
    file_path = os.path.join(path, file_name)
    item_ = read_annotations(file_path)
    location_class.append(item_[1])
    rot_class.append(item_[2])
    indicator += 1
print(f"indicator:{indicator}")
for i in range(9):
    count = location_class.count(str(i))
    print(f"location: {i}   count:{count}")

for j in range(12):
    angle_sum = []
    for k in range(30):
        angle = j*30 + k
        count_ = rot_class.count(str(angle))
        angle_sum.append(count_)
    print(f"angle class:{j}  num:{sum(angle_sum)}")





