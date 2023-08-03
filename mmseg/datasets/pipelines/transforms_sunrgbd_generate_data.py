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
import os
import imageio


@PIPELINES.register_module()
class RotateCircleGenerateData(object):
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
                 radius_ratio=1.0,
                 prepare_for_moco=False,
                 save_augmented=False):

        assert not (grid and random_grid), \
            'gird and random_grid cannot be setting at the same time'
        self.num = 0
        self.resize = (resize, resize)
        self.resize_for_loss = (resize//4, resize//4)  # for shrinking the image, 可能需要修改
        # self.radius = radius
        assert local_or_global in ['local', 'global'], \
            'local_or_global can only be local or global'
        self.local_or_global = local_or_global
        self.angle = angle
        # angle 部分的初始化
        angle_section = 360 / angle
        self.angle_list = [i * angle_section for i in range(angle)]
        # grid 部分的初始化
        self.grid = int(np.sqrt(grid))
        self.grid_section = resize // self.grid
        self.grid_location = []
        for i in range(self.grid):  # 生成每个box的左上角和右下角坐标
            for j in range(self.grid):
                top_left_bottom_right = [[j * self.grid_section, i * self.grid_section],
                                         [(j + 1) * self.grid_section, (i + 1) * self.grid_section]]
                self.grid_location.append(top_left_bottom_right)

        self.radius = int(self.grid_section // 2 * radius_ratio)
        self.prepare_for_moco = prepare_for_moco
        self.save_augmented = save_augmented

    # _resize_img 和 _resize_seg 相当于将图片和depth给缩放指定的尺寸
    def _resize_img(self, results, resize):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            results['img'], resize, return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img  # 这一步下面，img相关信息开始更新
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor  # 调整的scale信息

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

    def _resize_seg_beyond255(self, gt_semantic_seg, resize):
        """Resize semantic segmentation map with ``results['scale']``.
        这个方法适用于 original depth，此时depth的取值范围0-50000之间，
        而cv2.resize 只能处理0-255之间的数据值，所以这里需要根据需要重新进行处理
        """

        m = torch.nn.Upsample(size=resize)
        # gt_seg = m(torch.from_numpy(results['gt_semantic_seg'].astype(np.float)).unsqueeze(0).unsqueeze(0))  # 扩充维度
        gt_seg = m(torch.from_numpy(gt_semantic_seg.astype(np.float)).unsqueeze(0).unsqueeze(0))  # 扩充维度
        gt_seg = m(gt_seg).numpy().astype(np.int).squeeze(0).squeeze(0)

        return gt_seg

    def _rotate_seg_beyond255(self, seg, theta):
        """
        对于seg而言，还是因为超出了255，所以需要用其他方法实现
        :param results:
            theta: 这里输入的是360°之类的角度，应该将其转换成弧度
        :param theta:
        :return:
        """
        theta = torch.tensor(theta / 180 * math.pi)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                [torch.sin(theta), torch.cos(theta), 0]])
        grid = torch.nn.functional.affine_grid(rot_mat.unsqueeze(0),
                                               torch.from_numpy(seg.astype(np.float)).float().unsqueeze(
                                                   0).unsqueeze(0).size(), align_corners=True)
        rotate_seg = torch.nn.functional.grid_sample(torch.from_numpy(seg.astype(np.float)).float().unsqueeze(0).unsqueeze(0),
                                                     grid, align_corners=True)
        rotate_seg = rotate_seg.numpy().astype(np.int).squeeze(0).squeeze(0)
        return rotate_seg

    def _rotate_img_and_seg(self, results, paste_location1, paste_location2):
        # 之所以用
        # degrees = [0., 90., 180., 270.]
        degree_index1 = random.randint(0, self.angle)
        degree_index2 = random.randint(0, self.angle)  # degree_index1 和 degree_index2 可能一样
        degree1 = self.angle_list[degree_index1]
        degree2 = self.angle_list[degree_index2]
        results['sunrgbd_rotation1'] = degree_index1  # 将旋转角度索引存入
        results['sunrgbd_rotation2'] = degree_index2  # 将旋转角度索引存入

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
            img_resize = mmcv.imresize(results['img'], (self.grid_section, self.grid_section))  # 目标输出75*75
            img_crop1 = mmcv.imrotate(img_resize, degree1)
            # img_crop2 = mmcv.imresize(results['img'], (self.grid_section, self.grid_section))
            img_crop2 = mmcv.imrotate(img_resize, degree2)

            seg_resize = self._resize_seg_beyond255(results['gt_semantic_seg'], self.grid_section)
            seg_crop1 = self._rotate_seg_beyond255(seg_resize, degree1)  # degree1 是角度360，不是弧度
            # seg_crop2 = self._resize_seg_beyond255(results['gt_semantic_seg'], self.grid_section)
            seg_crop2 = self._rotate_seg_beyond255(seg_resize, degree2)

        # time_sticker = time.time()  # 保存相关图片
        # img_crop_save1 = mmcv.imwrite(img_crop1, "newback/imgs_segs/imgs/img_crop1_" + str(int(time_sticker)) + ".jpg")
        # seg_crop_save1 = mmcv.imwrite((seg_crop1 * 10000).astype(np.uint8),
        #                              "newback/imgs_segs/segs/seg_crop1_" + str(int(time_sticker)) + ".jpg")
        # img_crop_save2 = mmcv.imwrite(img_crop2, "newback/imgs_segs/imgs/img_crop2_" + str(int(time_sticker)) + ".jpg")
        # seg_crop_save2 = mmcv.imwrite((seg_crop2 * 10000).astype(np.uint8),
        #                              "newback/imgs_segs/segs/seg_crop2_" + str(int(time_sticker)) + ".jpg")

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
        ori_img1 = results['img'].copy()
        ori_seg1 = results['gt_semantic_seg'].copy()
        ori_img2 = results['img'].copy()
        ori_seg2 = results['gt_semantic_seg'].copy()
        img_crop_rotated1 = results['img_crop_rotated1']
        seg_crop_rotated1 = results['seg_crop_rotated1']
        img_crop_rotated2 = results['img_crop_rotated2']
        seg_crop_rotated2 = results['seg_crop_rotated2']

        # 对图片的第一个crop拼接操作
        for i in range(img_crop_rotated1.shape[0] - 1):
            for j in range(img_crop_rotated1.shape[1] - 1):
                if (i - self.radius) ** 2 + (j - self.radius) ** 2 <= self.radius ** 2:
                    ori_img1[j + paste_location1[0][1]][i + paste_location1[0][0]] = img_crop_rotated1[j][
                        i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                    ori_seg1[j + paste_location1[0][1]][i + paste_location1[0][0]] = seg_crop_rotated1[j][i]

        for i in range(img_crop_rotated2.shape[0] - 1):
            for j in range(img_crop_rotated2.shape[1] - 1):
                if (i - self.radius) ** 2 + (j - self.radius) ** 2 <= self.radius ** 2:
                    ori_img2[j + paste_location2[0][1]][i + paste_location2[0][0]] = img_crop_rotated2[j][
                        i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                    ori_seg2[j + paste_location2[0][1]][i + paste_location2[0][0]] = seg_crop_rotated2[j][i]

        # time_sticker = time.time()  # 保存相关图片
        if self.save_augmented:
            ori_img1_save = mmcv.imwrite(ori_img1, "newback/data/sunrgbd_generate/images/" + label_name)
            self._create_str_to_txt(results, label_name)
            # ori_img2_save = mmcv.imwrite(ori_img2, "newback/imgs_segs/imgs/ori_img2_" + "loc"+str(results['paste_location2']) + "_rot"+str(results['sunrgbd_rotation2']) + "_" + label_name)

            # seg_crop_save1 = mmcv.imwrite(ori_seg1,
            #                               "newback/data/sunrgbd_generate/depth/" + str.split(label_name,'.')[0]+'.png')
            imageio.imwrite("newback/data/sunrgbd_generate/depth/depth" + str.split(label_name, '.')[0]+'.png', ori_seg1.astype(np.uint16))
            # seg_crop_save2 = mmcv.imwrite((ori_seg2 * 10000).astype(np.uint8),
            #                               "newback/imgs_segs/segs/ori_seg2_" + "loc"+str(results['paste_location2']) + "_rot"+str(results['sunrgbd_rotation2']) + "_" + label_name)

            self.num = self.num + 1
            print(f"saved number:{self.num}")

        results['img_aug1'] = ori_img1
        results['gt_semantic_seg_aug1'] = ori_seg1
        results['img_aug2'] = ori_img2
        results['gt_semantic_seg_aug2'] = ori_seg2

    def _create_str_to_txt(self, results, label_name):
        """
        创建txt，并且写入
        """
        str_loc = str(results['paste_location1'])
        str_rot = str(results['sunrgbd_rotation1'])
        loc_detail = results['location_detail1']
        loc_0 = str(loc_detail[0][0])
        loc_1 = str(loc_detail[0][1])
        loc_2 = str(loc_detail[1][0])
        loc_3 = str(loc_detail[1][1])
        path_file_name = 'newback/data/sunrgbd_generate/labels/{}'.format(str.split(label_name, '.')[0]+'.txt')
        if not os.path.exists(path_file_name):
            with open(path_file_name, "w") as f:
                print(f)
        str_all = label_name + ' ' + str_loc + ' ' + str_rot + ' ' + loc_0 + ' ' + loc_1 + ' ' + loc_2 + ' ' + loc_3
        with open(path_file_name, "a") as f:
            f.write(str_all)

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
        location1 = random.randint(0, self.grid ** 2)  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
        location2 = random.randint(0, self.grid ** 2)
        results['paste_location1'] = location1  # 存储位置信息
        results['paste_location2'] = location2
        paste_location1 = self.grid_location[location1]
        paste_location2 = self.grid_location[location2]
        results['location_detail1'] = paste_location1
        results['location_detail2'] = paste_location2

        self._resize_img(results, self.resize)  # 将图片进行缩放
        gt_seg_resize = self._resize_seg_beyond255(results["gt_semantic_seg"], self.resize)  # 将depth压缩至255
        results["gt_semantic_seg"] = gt_seg_resize  # 这个就是原图进行压缩

        '''·
            # 如果 使用正方形九宫格旋转填充
            self._global_rotate_img_and_seg(results)
            self._paste_img_and_seg(results, paste_location)
        '''
        # 如果使用圆形旋转九宫格填充
        self._rotate_img_and_seg(results, paste_location1, paste_location2)
        self._paste_img_and_seg(results, paste_location1, paste_location2, label_name)

        results['img_aug1_for_loss'] = self._resize_img_for_loss_computation(results['img_aug1'], self.resize_for_loss)
        results['img_aug2_for_loss'] = self._resize_img_for_loss_computation(results['img_aug2'], self.resize_for_loss)
        results['gt_semantic_seg_aug1_for_loss'] = self._resize_seg_beyond255(results['gt_semantic_seg_aug1'],
                                                                              self.resize_for_loss)
        results['gt_semantic_seg_aug2_for_loss'] = self._resize_seg_beyond255(results['gt_semantic_seg_aug2'],
                                                                              self.resize_for_loss)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


