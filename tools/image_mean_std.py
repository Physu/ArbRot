'''
qhy
2018.12.3
'''
import os
import numpy as np
import cv2
# 图像数据集的路径
ims_path = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/image/'
# ims_list = os.listdir(ims_path)  # 直接读取文件夹下所有的图片信息
# 文件名汇总
train_list = []
val_list = []
total_list = []
txt_train = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/train_data_idx_backup.txt'
txt_val = '/mnt/disk7/lhy/Cascade/mmsegmentation-0.14.1/data/sunrgbd/sunrgbd_trainval/val_data_idx_backup.txt'
with open(txt_val, "r") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n')+".jpg"  # 去掉列表中每一个元素的换行符
        val_list.append(line)
        total_list.append(line)
        # print(line)

print("*"*30)
print(f"total val images:{len(val_list)}")

with open(txt_train, "r") as f:  # 打开文件
    for line in f.readlines():
        line = line.strip('\n')+".jpg"  # 去掉列表中每一个元素的换行符
        train_list.append(line)
        total_list.append(line)
        # print(line)

print("*"*30)
print(f"total train images:{len(train_list)}")
print(f"total images: {len(total_list)}")

def cal_mean_std(ims_list, data_type="val"):


    R_means = []
    G_means = []
    B_means = []
    for index, im_list in enumerate(ims_list):
        im = cv2.imread(ims_path + im_list)  # 读入的数据格式为BGR格式
        # extrect value of diffient channel
        im_B = im[:, :, 0]
        im_G = im[:, :, 1]
        im_R = im[:, :, 2]
        # count mean for every channel
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        # save single mean value to a set of means
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        # print('index:{} 图片：{} 的 RGB平均值为 \n[{}，{}，{}]'.format(index, im_list, im_R_mean, im_G_mean, im_B_mean))
    # three sets  into a large set
    a = [R_means, G_means, B_means]
    mean = [0, 0, 0]
    # count the sum of different channel means
    mean[0] = np.mean(a[0])  # R
    mean[1] = np.mean(a[1])  # G
    mean[2] = np.mean(a[2])  # B
    print("#"*30+data_type+"#"*30)
    print('数据集的RGB平均值为\n[{}，{}，{}]'.format(mean[0], mean[1], mean[2]))


    R_channel = 0
    G_channel = 0
    B_channel = 0
    for index, im_list in enumerate(ims_list):
        img = cv2.imread(ims_path + im_list)  # 读入的数据格式为BGR格式
        wh = img.shape[0] * img.shape[1]
        B_channel = B_channel + np.sum(np.power(img[:, :, 0] - mean[2], 2))/wh  # 之所以写成这样，是因为mean[2] 对应的是B, 而imread读入的顺序为BGR
        G_channel = G_channel + np.sum(np.power(img[:, :, 1] - mean[1], 2))/wh
        R_channel = R_channel + np.sum(np.power(img[:, :, 2] - mean[0], 2))/wh

    R_std = np.sqrt(R_channel / (index+1))
    G_std = np.sqrt(G_channel / (index+1))
    B_std = np.sqrt(B_channel / (index+1))

    print("total is %d R_std is %f, G_std is %f, B_std is %f" % (index+1, R_std, G_std, B_std))
    print("#" * 60)

cal_mean_std(val_list, data_type="val")
cal_mean_std(train_list, data_type="train")
cal_mean_std(total_list, data_type="total")


