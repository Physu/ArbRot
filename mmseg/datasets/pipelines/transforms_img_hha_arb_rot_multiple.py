import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES
import time
from PIL import Image
from PIL import Image, ImageFilter
from torchvision import transforms as _transforms
from mmseg.utils import build_from_cfg
import inspect
import torch
from mmcv.image.photometric import adjust_brightness, adjust_color, adjust_contrast, adjust_lighting
import scipy
import math
import cv2
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from mmseg.utils import collect_env, get_root_logger, traverse_replace
import os.path as osp


@PIPELINES.register_module()
class RotateCircleImgAndHHAArbRotMultiple(object):
    """Resize and rotate the image & seg.、
    不同于之前的方法：随机旋转，缩放，然后根据缩放的大小，在原始图片中随机贴回
    产生的标注：旋转的角度，缩放的大小，坐标信息（左上角，右下角）
    说白了，就是可以无限旋转然后贴回
    注意这个方法和之前的有很大的不同，在于这个位置不是random grid产生的了，而是随机贴回

    Args:

    """

    """Resize and rotate the image & seg.、
       这个方法只是对一张图片和对应的深度图进行处理，

       Args:

       """

    def __init__(self,
                 resize,
                 angle=4,
                 grid=None,
                 random_grid=None,
                 local_or_global='local',
                 loc_specify=None,
                 radius_ratio=1.0,
                 prepare_for_moco=False,
                 condidate_num=10,
                 save_augmented=False,
                 log_dir=None):

        assert not (grid and random_grid), \
            'gird and random_grid cannot be setting at the same time'
        self.num = 0
        self.resize = (resize, resize)
        self.resize_for_loss = (resize // 4, resize // 4)  # for shrinking the image, 可能需要修改
        # self.radius = radius
        assert local_or_global in ['local', 'global'], \
            'local_or_global can only be local or global'
        self.local_or_global = local_or_global
        if angle == 0:
            self.angle_list = [0]
            self.angle = 0
        elif angle is None:
            self.angle = None
            angle_section = 1
            self.angle_list = [i * angle_section for i in range(360)]
        else:
            self.angle = angle
            # angle 部分的初始化
            angle_section = 360 / angle
            self.angle_list = [i * angle_section for i in range(angle)]

        self.prepare_for_moco = prepare_for_moco
        self.candidate_num = condidate_num
        self.save_augmented = save_augmented
        if log_dir is not None:
            self.logger = get_root_logger(log_level='INFO')
        self.loc_specify = loc_specify

    # _resize_img 和 _resize_seg 相当于将图片和depth给缩放指定的尺寸
    def _resize_img_and_dep(self, results, resize):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            results['img'], resize, return_scale=True)
        gt_semantic_seg, _, _ = mmcv.imresize(
            results['gt_semantic_seg'], resize, return_scale=True)

        if 'ori_img' in results:
            results['ori_img'], _, _ = mmcv.imresize(
                results['ori_img'], resize, return_scale=True)
        if 'ori_gt_semantic_seg' in results:
            results['ori_gt_semantic_seg'], _, _ = mmcv.imresize(
                results['ori_gt_semantic_seg'], resize, return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img  # 这一步下面，img相关信息开始更新
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor  # 调整的scale信息
        # seg 信息更新
        results['gt_semantic_seg'] = gt_semantic_seg

    # _resize_img 和 _resize_seg 相当于将图片和depth给缩放指定的尺寸
    def _resize_img_for_loss_computation(self, img, resize):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            img, resize, return_scale=True)

        return img  # 这一步下面，img相关信息开始更新

    def _resize_seg(self, results):
        """ 这个适用于数值在0-255区间的，对于深度信息就不太适用了
        Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            gt_seg = mmcv.imresize(
                results[key], self.resize)
            # results[key] = gt_seg  # 这一步屏蔽，统一放到call 中处理
        return gt_seg

    def _rotate_img_and_seg(self, results, paste_location1, paste_location2, degree1, degree2):
        # 之所以用
        # degrees = [0., 90., 180., 270.]

        # degree = 180  # Rotation angle in degrees, positive values mean clockwise rotation
        # results['sunrgbd_rotation'] = 2
        box1 = np.array([paste_location1[0][0], paste_location1[0][1], paste_location1[1][0],
                         paste_location1[1][1]])
        box2 = np.array([paste_location2[0][0], paste_location2[0][1], paste_location2[1][0],
                         paste_location2[1][1]])

        if self.local_or_global == 'local':  # 基本不用这个，目标是构建一种局部和总体之间的关系
            img_crop1 = mmcv.imcrop(results['img'], box1)
            img_crop1 = mmcv.imrotate(img_crop1, degree1)
            img_crop2 = mmcv.imcrop(results['img'], box2)
            img_crop2 = mmcv.imrotate(img_crop2, degree2)

            seg_crop1 = mmcv.imcrop(results['gt_semantic_seg'], box1)
            seg_crop1 = mmcv.imrotate(seg_crop1, degree1)
            seg_crop2 = mmcv.imcrop(results['gt_semantic_seg'], box2)
            seg_crop2 = mmcv.imrotate(seg_crop2, degree2)

        elif self.local_or_global == 'global':
            img_resize1 = mmcv.imresize(results['img'], (results['radius1'] * 2, results['radius1'] * 2))
            img_resize2 = mmcv.imresize(results['img'], (results['radius2'] * 2, results['radius2'] * 2))

            seg_resize1 = mmcv.imresize(results['gt_semantic_seg'], (results['radius1'] * 2, results['radius1'] * 2))
            seg_resize2 = mmcv.imresize(results['gt_semantic_seg'], (results['radius2'] * 2, results['radius2'] * 2))
            # 目标输出75*75
            # 对图片的第一个crop拼接操作
            # for i in range(img_resize1.shape[0]):
            #     for j in range(img_resize1.shape[1]):
            #         if (i - results['radius1']) ** 2 + (j - results['radius1']) ** 2 > results['radius1'] ** 2:
            #             img_resize1[i][j] = np.array([0, 0, 0])  # 注意这个ji顺序，否则会出现对角线反转这种情况
            #             seg_resize1[i][j] = np.array([0, 0, 0])
            #
            # for i in range(img_resize2.shape[0] - 1):
            #     for j in range(img_resize2.shape[1] - 1):
            #         if (i - results['radius2']) ** 2 + (j - results['radius2']) ** 2 > results['radius2'] ** 2:
            #             img_resize2[i][j] = np.array([0, 0, 0])  # 注意这个ji顺序，否则会出现对角线反转这种情况
            #             seg_resize2[i][j] = np.array([0, 0, 0])

            img_crop1 = mmcv.imrotate(img_resize1, degree1)
            img_crop2 = mmcv.imrotate(img_resize2, degree2)

            seg_crop1 = mmcv.imrotate(seg_resize1, degree1)
            seg_crop2 = mmcv.imrotate(seg_resize2, degree2)

            # time_sticker = time.time()  # 保存相关图片
            # img_crop_save1 = mmcv.imwrite(img_crop1, "newback/imgs_segs/ArbRotPlus/test_samples/0323_imgcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop1.astype(np.uint8), "newback/imgs_segs/ArbRotPlus/test_samples/0323_segcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['ann']['seg_map'])
            #
            # img_crop_save1 = mmcv.imwrite(img_crop2, "newback/imgs_segs/ArbRotPlus/test_samples/0323_imgcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop2.astype(np.uint8), "newback/imgs_segs/ArbRotPlus/test_samples/0323_segcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['ann']['seg_map'])

        results['img_crop_rotated1'] = img_crop1  # 存储对应的小图片和深度信息 75*75
        results['seg_crop_rotated1'] = seg_crop1
        results['img_crop_rotated2'] = img_crop2
        results['seg_crop_rotated2'] = seg_crop2

    def _paste_img_and_seg(self, results, paste_location1, paste_location2, label_name):
        '''
        :param results:
        :param paste_location1:
        :param paste_location2:
        :param label_name:
        :return:

        python里numpy默认的是浅拷贝，即拷贝的是对象的地址，结果是修改拷贝的值的时候原对象也会随之改变，
        narray.copy()进行深拷贝，即拷贝numpy对象中的数据，而不是地址
        '''
        ori_img1 = results['img_aug1'].copy()
        ori_seg1 = results['gt_semantic_seg_aug1'].copy()
        ori_img2 = results['img_aug1'].copy()
        ori_seg2 = results['gt_semantic_seg_aug2'].copy()
        img_crop_rotated1 = results['img_crop_rotated1']
        seg_crop_rotated1 = results['seg_crop_rotated1']
        img_crop_rotated2 = results['img_crop_rotated2']
        seg_crop_rotated2 = results['seg_crop_rotated2']

        # 对图片的第一个crop拼接操作
        for i in range(img_crop_rotated1.shape[0]):
            for j in range(img_crop_rotated1.shape[1]):
                if (i - results['radius1']) ** 2 + (j - results['radius1']) ** 2 < results['radius1'] ** 2:
                    ori_img1[j + paste_location1[0][1]][i + paste_location1[0][0]] = img_crop_rotated1[j][
                        i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                    ori_seg1[j + paste_location1[0][1]][i + paste_location1[0][0]] = seg_crop_rotated1[j][i]

        for i in range(img_crop_rotated2.shape[0] - 1):
            for j in range(img_crop_rotated2.shape[1] - 1):
                if (i - results['radius2']) ** 2 + (j - results['radius2']) ** 2 <= results['radius2'] ** 2:
                    ori_img2[j + paste_location2[0][1]][i + paste_location2[0][0]] = img_crop_rotated2[j][
                        i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                    ori_seg2[j + paste_location2[0][1]][i + paste_location2[0][0]] = seg_crop_rotated2[j][i]

        # time_sticker = time.time()  # 保存相关图片
        if self.save_augmented:
            ori_img1_save = mmcv.imwrite(ori_img1, "newback/imgs_segs/ArbRotPlus/test_samples/ori_img1_" + "loc" + str(
                results['paste_location1']) + "_rot" + str(results['sunrgbd_rotation1']) + "_" + label_name)
            ori_img2_save = mmcv.imwrite(ori_img2, "newback/imgs_segs/ArbRotPlus/test_samples/ori_img2_" + "loc" + str(
                results['paste_location2']) + "_rot" + str(results['sunrgbd_rotation2']) + "_" + label_name)

            seg_crop_save1 = mmcv.imwrite(ori_seg1,
                                          "newback/imgs_segs/ArbRotPlus/test_samples/ori_seg1_" + "loc" + str(
                                              results['paste_location1']) + "_rot" + str(
                                              results['sunrgbd_rotation1']) + "_" + label_name)
            seg_crop_save2 = mmcv.imwrite(ori_seg2,
                                          "newback/imgs_segs/ArbRotPlus/test_samples/ori_seg2_" + "loc" + str(
                                              results['paste_location2']) + "_rot" + str(
                                              results['sunrgbd_rotation2']) + "_" + label_name)
            print(f"saved number:{self.num}")
            self.num = self.num + 1

        results['img_aug1'] = ori_img1
        results['gt_semantic_seg_aug1'] = ori_seg1
        results['img_aug2'] = ori_img2
        results['gt_semantic_seg_aug2'] = ori_seg2

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.
        关于旋转角度： 顺时针转动
        位置： 从左到右，从上到下

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.

            |--------x
            |
            |
            y

        """
        # locations = [[[0, 0], [75, 75]], [[75, 0], [150, 75]], [[150, 0], [225, 75]],
        #              [[0, 75], [75, 150]], [[75, 75], [150, 150]], [[150, 75], [225, 150]],
        #              [[0, 150], [75, 225]], [[75, 150], [150, 225]], [[150, 150], [225, 225]]]
        label_name = results['img_info']['filename']

        self._resize_img_and_dep(results, self.resize)  # 将图片和深度进行缩放
        results['img_aug1'] = results['img'].copy()
        results['gt_semantic_seg_aug1'] = results['gt_semantic_seg'].copy()
        results['img_aug2'] = results['img'].copy()
        results['gt_semantic_seg_aug2'] = results['gt_semantic_seg'].copy()

        aug1_list = []
        aug2_list = []
        paste_location1_list = []
        paste_location2_list = []
        rot1_list = []
        rot2_list = []
        for i in range(np.random.randint(low=1, high=self.candidate_num)):
            scale1 = np.round(np.random.uniform(low=float(0.1), high=float(0.9), size=1), 2)
            scale2 = np.round(np.random.uniform(low=float(0.1), high=float(0.9), size=1), 2)
            radius1 = int(self.resize[0] * scale1 / 2)  #
            radius2 = int(self.resize[0] * scale2 / 2)

            # area1 = math.pi * pow(radius1, 2)
            # area2 = math.pi * pow(radius2, 2)

            aug1_list.append(radius1)
            aug2_list.append(radius2)

        aug1_list = sorted(aug1_list, reverse=True)  # 降序排序
        aug2_list = sorted(aug2_list, reverse=True)

        for radius1, radius2 in zip(aug1_list, aug2_list):
            if self.loc_specify is not None:
                # 中心点贴回
                paste_location1 = np.array([[int(self.resize[0]/2 - radius1), int(self.resize[0]/2 - radius1)],
                                            [int(self.resize[0]/2 + radius1), int(self.resize[0]/2 + radius1)]])

                paste_location2 = np.array([[int(self.resize[0]/2 - radius2), int(self.resize[0]/2 - radius2)],
                                            [int(self.resize[0]/2 + radius2), int(self.resize[0]/2 + radius2)]])
            else:
                interval1 = np.random.randint(low=float(radius1), high=float(self.resize[0] - radius1),
                                                        size=2)  # 返回的表示中心点的坐标

                interval2 = np.random.randint(low=float(radius2), high=float(self.resize[0] - radius2),
                                                        size=2) # 返回的表示中心点的坐标

                paste_location1 = np.array([[int(interval1[0] - radius1), int(interval1[1] - radius1)],
                                            [int(interval1[0] + radius1), int(interval1[1] + radius1)]])

                paste_location2 = np.array([[int(interval2[0] - radius2), int(interval2[1] - radius2)],
                                            [int(interval2[0] + radius2), int(interval2[1] + radius2)]])

                results['paste_location1'] = paste_location1  # 存储位置信息
                results['paste_location2'] = paste_location2

                paste_location1_list.append(paste_location1)
                paste_location2_list.append(paste_location2)

                results['radius1'] = radius1
                results['radius2'] = radius2


            '''·
                # 如果 使用正方形九宫格旋转填充
                self._global_rotate_img_and_seg(results)
                self._paste_img_and_seg(results, paste_location)
            '''

            if self.angle == 0:
                degree_index1 = 0
                degree_index2 = 0
                degree1 = 0
                degree2 = 0
            elif self.angle is None:
                degree_index1 = random.randint(0, 360)
                degree_index2 = random.randint(0, 360)
                degree1 = self.angle_list[degree_index1]
                degree2 = self.angle_list[degree_index2]
            else:
                degree_index1 = random.randint(0, self.angle)
                degree_index2 = random.randint(0, self.angle)  # degree_index1 和 degree_index2 可能一样
                degree1 = self.angle_list[degree_index1]
                degree2 = self.angle_list[degree_index2]
            results['sunrgbd_rotation1'] = degree_index1  # 将旋转角度索引存入
            results['sunrgbd_rotation2'] = degree_index2  # 将旋转角度索引存入

            rot1_list.append(degree1)
            rot2_list.append(degree2)
            # 如果使用圆形旋转九宫格填充
            self._rotate_img_and_seg(results, paste_location1, paste_location2, degree1, degree2)  # 传入paste_location 是为了local缩放贴回时需要的信息
            self._paste_img_and_seg(results, paste_location1, paste_location2, label_name)


        # ori_img1_save = mmcv.imwrite(results['img_aug1'],
        #                              "newback/imgs_segs/arbrot_multiple/img/ori_img1_" + "loc" + str(
        #                                  results['paste_location1']) + "_rot" + str(
        #                                  results['sunrgbd_rotation1']) + "_" + label_name)
        # seg_crop_save1 = mmcv.imwrite(results['gt_semantic_seg_aug1'],
        #                               "newback/imgs_segs/arbrot_multiple/dep/ori_seg1_" + "loc" + str(
        #                                   results['paste_location1']) + "_rot" + str(
        #                                   results['sunrgbd_rotation1']) + "_" + label_name)

        if np.isnan(results['img_aug1']).any() or np.isnan(results['img_aug2']).any():
            self.logger.info('Transforms_Img_HHA info:\n' + 'the img_aug exists problem!' + '\n')

        # results['paste_location1'] = np.array([int(interval1[0] - radius1), int(interval1[1] - radius1),
        #                                        int(interval1[0] + radius1), int(interval1[1] + radius1)])
        # results['paste_location2'] = np.array([int(interval2[0] - radius2), int(interval2[1] - radius2),
        #                                        int(interval2[0] + radius2), int(interval2[1] + radius2)])
        padding_len = self.candidate_num - len(rot1_list)
        # 旋转角度补361，表示这个地方是不需要角度预测的
        results['sunrgbd_rotation1'] = np.array(np.pad(rot1_list, (0, padding_len), 'constant', constant_values=(0, 361)))  # 将旋转角度索引存入
        results['sunrgbd_rotation2'] = np.array(np.pad(rot2_list, (0, padding_len), 'constant', constant_values=(0, 361)))
        bb = np.zeros((padding_len, 2, 2))
        results['paste_location1'] = np.array(np.concatenate((paste_location1_list, bb), axis=0))  # 存储位置信息
        results['paste_location2'] = np.array(np.concatenate((paste_location2_list, bb), axis=0))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'