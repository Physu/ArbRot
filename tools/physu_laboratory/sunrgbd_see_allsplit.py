# coding:UTF-8

import scipy.io as scio
import os
import shutil

dataFile = './tools/physu_laboratory/allsplit.mat'
source_location = '/mnt/disk7/lhy/RGBxD/sun_rgbd/sunrgbd_hha'  # 注意这里，文件夹下有两个文件夹：hha, hha_bfx
destination_location = '/mnt/disk6/data/sunrgbd/sunrgbd_trainval'
data = scio.loadmat(dataFile)
i = 5050
for value in data['alltrain'][0]:
    i = i+1
    print(f'index:{i}, value:{value[0][24:]}')
    source_location_hha = source_location + value[0][24:] +'/hha/'
    source_location_hha_bfx = source_location + value[0][24:] + '/hha_bfx/'
    file_hha = os.listdir(source_location_hha)
    file_hha_bfx = os.listdir(source_location_hha_bfx)
    source_location_hha_ = source_location_hha + file_hha[0]
    source_location_hha_bfx_ = source_location_hha_bfx + file_hha_bfx[0]

    idx_str = f'{i:06d}'

    destination_location_hha = destination_location + '/hha/' + idx_str + '.png'
    destination_location_hha_bfx = destination_location +'/hha_bfx/' + idx_str + '.png'
    print(f'index:{i}, source_location_hha_:{source_location_hha_}')

    shutil.copyfile(source_location_hha_, destination_location_hha)
    shutil.copyfile(source_location_hha_bfx_, destination_location_hha_bfx)
    # if i >10:
    #     break

print(f"finished")

