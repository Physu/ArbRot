'''
qhy
2018.12.3
modifier by physu
2021.10.27
'''
import os
import numpy as np
import cv2
# 图像数据集的路径
ims_path = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/depth_bfx_png/'
# ims_list = os.listdir(ims_path)  # 直接读取文件夹下所有的图片信息
# 文件名汇总
train_list = []
val_list = []
total_list = []
txt_train = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/train_data_idx_backup.txt'
txt_val = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/val_data_idx_backup.txt'

with open(txt_val, "r") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n')+".png"  # 去掉列表中每一个元素的换行符
        val_list.append(line)
        total_list.append(line)
        # print(line)

print("*"*30)
print(f"total val labels:{len(val_list)}")
#
# with open(txt_train, "r") as f:  # 打开文件
#     for line in f.readlines():
#         line = line.strip('\n')+".png"  # 去掉列表中每一个元素的换行符
#         train_list.append(line)
#         total_list.append(line)
#         # print(line)
#
# print("*"*30)
# print(f"total train labels:{len(train_list)}")
# print(f"total labels: {len(total_list)}")

def cal_mean_std(ims_list, data_type="val"):


    means = []

    for index, im_list in enumerate(ims_list):
        im = cv2.imread(ims_path + im_list)  # 读入的数据为单通道
        # extrect value of diffient channel

        # count mean for every channel
        im_R_mean = np.mean(im)

        # save single mean value to a set of means
        means.append(im_R_mean)

        # print('index:{} 图片：{} 的 RGB平均值为 \n[{}，{}，{}]'.format(index, im_list, im_R_mean, im_G_mean, im_B_mean))

    print("#"*30+data_type+"#"*30)
    print('数据集的RGB平均值为\n[{}，{}，{}]'.format(means))


    channel = 0

    for index, im_list in enumerate(ims_list):
        img = cv2.imread(ims_path + im_list)  # 读入的数据格式为BGR格式
        wh = img.shape[0] * img.shape[1]
        channel_mean = channel + np.sum(np.power(img[:, :, 0] - mean[2], 2))/wh
    channel_std = np.sqrt(channel / (index+1))


    print("total is %d R_std is %f, G_std is %f, B_std is %f" % (index+1, channel_std))
    print("#" * 60)

# cal_mean_std(val_list, data_type="val")
# cal_mean_std(train_list, data_type="train")
# cal_mean_std(total_list, data_type="total")


