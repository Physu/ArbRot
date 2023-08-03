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
import torchvision.transforms



@PIPELINES.register_module()
class RotateCircleImgAndHHA(object):
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
                 save_augmented=False,
                 log_dir=None):

        assert not (grid and random_grid), \
            'gird and random_grid cannot be setting at the same time'
        self.num = 0
        self.resize = (resize, resize)
        self.resize_for_loss = (resize//4, resize//4)  # for shrinking the image, 可能需要修改
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
        if log_dir is not None:
            self.logger = get_root_logger(log_level='INFO')
        self.loc_specify = loc_specify

    # _resize_img 和 _resize_seg 相当于将图片和depth给缩放指定的尺寸
    def _resize_img(self, results, resize):
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

    def _rotate_img_and_seg(self, results, paste_location1, paste_location2):
        # 之所以用
        # degrees = [0., 90., 180., 270.]
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
            img_resize = mmcv.imresize(results['img'], (self.grid_section-1, self.grid_section-1))  # 目标输出75*75
            img_crop1 = mmcv.imrotate(img_resize, degree1)
            img_crop2 = mmcv.imrotate(img_resize, degree2)
            seg_resize = mmcv.imresize(results['gt_semantic_seg'], (self.grid_section, self.grid_section))
            seg_crop1 = mmcv.imrotate(seg_resize, degree1)
            seg_crop2 = mmcv.imrotate(seg_resize, degree2)


            # time_sticker = time.time()  # 保存相关图片
            # img_crop_save1 = mmcv.imwrite(img_crop1, "newback/imgs_segs/imgs_head/0323_imgcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop1, "newback/imgs_segs/imgs_head/0323_segcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['ann'])
            #
            # img_crop_save1 = mmcv.imwrite(img_crop2, "newback/imgs_segs/imgs_head/0323_imgcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop2, "newback/imgs_segs/imgs_head/0323_segcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['ann'])


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
            ori_img1_save = mmcv.imwrite(ori_img1, "newback/imgs_segs/imgs/ori_img1_" + "loc"+str(results['paste_location1']) + "_rot"+str(results['sunrgbd_rotation1']) + "_" + label_name)
            ori_img2_save = mmcv.imwrite(ori_img2, "newback/imgs_segs/imgs/ori_img2_" + "loc"+str(results['paste_location2']) + "_rot"+str(results['sunrgbd_rotation2']) + "_" + label_name)

            seg_crop_save1 = mmcv.imwrite(ori_seg1,
                                          "newback/imgs_segs/segs/ori_seg1_" + "loc"+str(results['paste_location1']) + "_rot"+str(results['sunrgbd_rotation1']) + "_" + label_name)
            seg_crop_save2 = mmcv.imwrite(ori_seg2,
                                          "newback/imgs_segs/segs/ori_seg2_" + "loc"+str(results['paste_location2']) + "_rot"+str(results['sunrgbd_rotation2']) + "_" + label_name)
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
        if self.loc_specify is not None:
            location1 = self.loc_specify  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
            location2 = self.loc_specify
        else:
            location1 = random.randint(0, self.grid ** 2)  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
            location2 = random.randint(0, self.grid ** 2)
        results['paste_location1'] = location1  # 存储位置信息
        results['paste_location2'] = location2
        paste_location1 = self.grid_location[location1]
        paste_location2 = self.grid_location[location2]

        self._resize_img(results, self.resize)  # 将图片和深度进行缩放
        # gt_seg_resize = self._resize_seg_beyond255(results["gt_semantic_seg"], self.resize)  # 将depth压缩至255
        # results["gt_semantic_seg"] = gt_seg_resize  # 这个就是原图进行压缩

        '''·
            # 如果 使用正方形九宫格旋转填充
            self._global_rotate_img_and_seg(results)
            self._paste_img_and_seg(results, paste_location)
        '''
        # 如果使用圆形旋转九宫格填充
        self._rotate_img_and_seg(results, paste_location1, paste_location2)  # 传入paste_location 是为了local缩放贴回时需要的信息
        self._paste_img_and_seg(results, paste_location1, paste_location2, label_name)

        if np.isnan(results['img_aug1']).any() or np.isnan(results['img_aug2']).any():
            self.logger.info('Transforms_Img_HHA info:\n' + 'the img_aug exists problem!' + '\n')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class NormalizeImgAndHHA(object):
    """Normalize the image and label

    Added key is "img_norm_cfg and label_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, img_norm_cfg=None, with_arbrot2=True):
        self.img_norm_cfg = img_norm_cfg
        if self.img_norm_cfg is not None:
            self.img_mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
            self.img_std = np.array(img_norm_cfg['std'], dtype=np.float32)
            self.to_rgb = img_norm_cfg['to_rgb']
            self.with_arbrot2 = with_arbrot2

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.with_arbrot2:

            img_aug1 = results['img_aug1']
            img_aug2 = results['img_aug2']
            gt_semantic_seg_aug1 = results['gt_semantic_seg_aug1']
            gt_semantic_seg_aug2 = results['gt_semantic_seg_aug2']

            img_aug1 = F.to_tensor(img_aug1.copy()).float()  # 这一步相当于0-255，归一化到0-1.0
            img_aug2 = F.to_tensor(img_aug2.copy()).float()
            gt_semantic_seg_aug1 = F.to_tensor(gt_semantic_seg_aug1.astype(np.uint8)).float()
            gt_semantic_seg_aug2 = F.to_tensor(gt_semantic_seg_aug2.astype(np.uint8)).float()


            if self.img_norm_cfg is not None:
                img_aug1 = F.normalize(img_aug1, self.img_mean, self.img_std)  # 这一步相当于0-255，归一化到0-1.0
                img_aug2 = F.normalize(img_aug2, self.img_mean, self.img_std)
                gt_semantic_seg_aug1 = F.normalize(gt_semantic_seg_aug1, self.img_mean, self.img_std)
                gt_semantic_seg_aug2 = F.normalize(gt_semantic_seg_aug2, self.img_mean, self.img_std)
                if 'ori_img' in results:
                    ori_img = results['ori_img']
                    ori_img = F.to_tensor(ori_img.copy()).float()
                    ori_img = F.normalize(ori_img, self.img_mean, self.img_std)
                    results['ori_img'] = ori_img
                if 'ori_gt_semantic_seg' in results:
                    ori_gt_semantic_seg = results['ori_gt_semantic_seg']
                    ori_gt_semantic_seg = F.to_tensor(ori_gt_semantic_seg.astype(np.uint8)).float()
                    ori_gt_semantic_seg = F.normalize(ori_gt_semantic_seg, self.img_mean, self.img_std)
                    results['ori_gt_semantic_seg'] = ori_gt_semantic_seg

            results['img_aug1'] = img_aug1
            results['img_aug2'] = img_aug2
            results['gt_semantic_seg_aug1'] = gt_semantic_seg_aug1
            results['gt_semantic_seg_aug2'] = gt_semantic_seg_aug2
        else:
            img_aug1 = results['img']
            gt_semantic_seg_aug1 = results['gt_semantic_seg']

            img_aug1 = F.to_tensor(img_aug1.copy()).float()  # 这一步相当于0-255，归一化到0-1.0

            gt_semantic_seg_aug1 = F.to_tensor(gt_semantic_seg_aug1.astype(np.uint8)).float()

            if self.img_norm_cfg is not None:
                img_aug1 = F.normalize(img_aug1, self.img_mean, self.img_std)  # 这一步相当于0-255，归一化到0-1.0

                gt_semantic_seg_aug1 = F.normalize(gt_semantic_seg_aug1, self.img_mean, self.img_std)

                if 'ori_img' in results:
                    ori_img = results['ori_img']
                    ori_img = F.to_tensor(ori_img.copy()).float()
                    ori_img = F.normalize(ori_img, self.img_mean, self.img_std)
                    results['ori_img'] = ori_img
                if 'ori_gt_semantic_seg' in results:
                    ori_gt_semantic_seg = results['ori_gt_semantic_seg']
                    ori_gt_semantic_seg = F.to_tensor(ori_gt_semantic_seg.astype(np.uint8)).float()
                    ori_gt_semantic_seg = F.normalize(ori_gt_semantic_seg, self.img_mean, self.img_std)
                    results['ori_gt_semantic_seg'] = ori_gt_semantic_seg

            results['img_aug1'] = img_aug1
            results['img_aug2'] = img_aug1
            results['gt_semantic_seg_aug1'] = gt_semantic_seg_aug1
            results['gt_semantic_seg_aug2'] = gt_semantic_seg_aug1


        # ori_img1_save = mmcv.imwrite(img_aug1,
        #                              "newback/imgs_segs/imgs_beforenorm/img1_" + results['img_info']['filename'])
        # # ori_img2_save = mmcv.imwrite(img_aug2,
        # #                              "newback/imgs_segs/imgs_beforenorm/img2_" + results['img_info']['filename'])
        # if len(gt_semantic_seg_aug1.astype(np.uint8).shape) == 2 or gt_semantic_seg_aug1.astype(np.uint8).shape[-1] == 1:
        #     img = cv2.cvtColor(gt_semantic_seg_aug1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        #     ori_depth_save = mmcv.imwrite(img,
        #                                   "newback/imgs_segs/imgs_beforenorm/depth1_" + results['ann_info']['seg_map'])
        # 这个sava_image 保存图片顺序rgb
        # ori_depth_save = save_image(gt_semantic_seg_aug1.astype(np.uint8),
        #                              "newback/imgs_segs/debug_nan/depth1_" + results['ann_info']['seg_map'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomFlipImgHHA(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img_aug1'] = mmcv.imflip(
                results['img_aug1'], direction=results['flip_direction'])
            results['img_aug2'] = mmcv.imflip(
                results['img_aug2'], direction=results['flip_direction'])

            results['gt_semantic_seg_aug1'] = mmcv.imflip(
                results['gt_semantic_seg_aug1'], direction=results['flip_direction']).copy()
            results['gt_semantic_seg_aug2'] = mmcv.imflip(
                results['gt_semantic_seg_aug2'], direction=results['flip_direction']).copy()

            if 'ori_img' in results:
                results['ori_img'] = mmcv.imflip(results['ori_img'], direction=results['flip_direction'])
            if 'ori_gt_semantic_seg' in results:
                results['ori_gt_semantic_seg'] = mmcv.imflip(
                    results['ori_gt_semantic_seg'], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomCropImgHHA(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size=(500, 500), padding=0):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.padding = padding

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        # img = results['img']
        # img_aug1 = results['img_aug1']
        # img_aug2 = results['img_aug2']
        # gt_semantic_seg_aug1 = results['gt_semantic_seg_aug1']
        # gt_semantic_seg_aug2 = results['gt_semantic_seg_aug2']
        # crop_bbox = self.get_crop_bbox(img)
        crop_bbox_img = self.get_crop_bbox(results['img'])  # 这部分主要获得下一步需要的参数

        seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox_img)
        # crop the image
        img = self.crop(results['img'], crop_bbox_img)

        results['img'] = img
        results['img_shape'] = img.shape

        results['gt_semantic_seg'] = seg_temp
        results['gt_semantic_seg_shape'] = seg_temp.shape


        # ori_img1_save = mmcv.imwrite(img,
        #                              "newback/imgs_segs/augmentation_vis/img1_" + results['img_info']['filename'])
        # ori_img1_save = mmcv.imwrite(seg_temp.astype(np.uint8),
        #                              "newback/imgs_segs/augmentation_vis/dep1_" + results['img_info']['filename'])

        return results

@PIPELINES.register_module()
class ColorJitterImgHHA(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, results):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        # hue_factor = random.uniform(-self.hue, self.hue)

        img = results['img']
        dep = results['gt_semantic_seg']
        fn_idx = torch.randperm(3)  # 3种图片处理方式，随机采用的顺序不固定
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                img = adjust_brightness(img, brightness_factor)
                dep = adjust_brightness(dep, brightness_factor)
            elif fn_id == 1 and self.contrast is not None:
                img = adjust_contrast(img, contrast_factor)
                dep = adjust_contrast(dep, contrast_factor)
            elif fn_id == 2 and self.saturation is not None:
                img = adjust_color(img, saturation_factor)
                dep = adjust_color(dep, saturation_factor)
            # elif fn_id == 3 and self.hue is not None:
            #     img = F.adjust_hue(img, hue_factor)
            #     dep = F.adjust_hue(dep, hue_factor)

        results['img'] = img
        results['gt_semantic_seg'] = dep

        # ori_img1_save = mmcv.imwrite(img,
        #                              "newback/imgs_segs/augmentation_vis/img1_" + results['img_info']['filename'])
        # ori_img1_save = mmcv.imwrite(dep.astype(np.uint8),
        #                              "newback/imgs_segs/augmentation_vis/dep1_" + results['img_info']['filename'])

        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        # format_string += ', hue={0})'.format(self.hue)
        return format_string

@PIPELINES.register_module()
class RandomGrayscaleImgHHA(object):
    """Randomly convert image to grayscale with a probability of gray_prob.

    Args:
        gray_prob (float): Probability that image should be converted to
            grayscale. Default: 0.1.

    Returns:
        ndarray: Image after randomly grayscale transform.

    Notes:
        - If input image is 1 channel: grayscale version is 1 channel.
        - If input image is 3 channel: grayscale version is 3 channel
          with r == g == b.
    """

    def __init__(self, gray_prob=0.1):
        self.gray_prob = gray_prob

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.

        Returns:
            ndarray: Randomly grayscaled image.
        """

        img = results['img']
        dep = results['gt_semantic_seg']
        num_output_channels = img.shape[2]
        if random.random() < self.gray_prob:
            if num_output_channels > 1:
                img = mmcv.rgb2gray(img)[:, :, None]
                dep = mmcv.rgb2gray(dep)[:, :, None]
                results['img'] = np.dstack(
                    [img for _ in range(num_output_channels)])
                results['gt_semantic_seg'] = np.dstack(
                    [dep for _ in range(num_output_channels)])
        # ori_img1_save = mmcv.imwrite(img,
        #                              "newback/imgs_segs/augmentation_vis/img1_" + results['img_info']['filename'])
        # ori_img1_save = mmcv.imwrite(dep.astype(np.uint8),
        #                              "newback/imgs_segs/augmentation_vis/dep1_" + results['img_info']['filename'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gray_prob={self.gray_prob})'


@PIPELINES.register_module()
class GaussianBlurImgHHA(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max, p=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            dep = results['gt_semantic_seg']
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)

            img = Image.fromarray(img)  # ndarray 转换成为 PIL.Image 反之则numpy.array(img)：img对象转化为np数组
            dep = Image.fromarray(dep.astype(np.uint8))  # 这一步操作是因为dep是np.uint16格式的参数
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            dep = dep.filter(ImageFilter.GaussianBlur(radius=sigma))
            img = np.array(img)
            dep = np.array(dep)
            results['img'] = img
            results['gt_semantic_seg'] = dep
            # ori_img1_save = mmcv.imwrite(img,
            #                              "newback/imgs_segs/augmentation_vis/img1_" + results['img_info']['filename'])
            # ori_img1_save = mmcv.imwrite(dep.astype(np.uint8),
            #                              "newback/imgs_segs/augmentation_vis/dep1_" + results['img_info']['filename'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomFlipImgHHABefore(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            results['gt_semantic_seg'] = mmcv.imflip(
                results['gt_semantic_seg'], direction=results['flip_direction']).copy()

        # ori_img1_save = mmcv.imwrite(results['img'],
        #                              "newback/imgs_segs/augmentation_vis/img1_" + results['img_info']['filename'])
        # ori_img1_save = mmcv.imwrite(results['gt_semantic_seg'],
        #                              "newback/imgs_segs/augmentation_vis/dep1_" + results['img_info']['filename'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class RotateRectangleImgAndHHA(object):
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
        if log_dir is not None:
            self.logger = get_root_logger(log_level='INFO')
        self.loc_specify = loc_specify

    # _resize_img 和 _resize_seg 相当于将图片和depth给缩放指定的尺寸
    def _resize_img(self, results, resize):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            results['img'], resize, return_scale=True)
        gt_semantic_seg, _, _ = mmcv.imresize(
            results['gt_semantic_seg'], resize, return_scale=True)
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

    def _rotate_img_and_seg(self, results, paste_location1, paste_location2):
        # 之所以用
        # degrees = [0., 90., 180., 270.]
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
            img_resize = mmcv.imresize(results['img'], (self.grid_section - 1, self.grid_section - 1))  # 目标输出75*75
            img_crop1 = mmcv.imrotate(img_resize, degree1)
            img_crop2 = mmcv.imrotate(img_resize, degree2)
            seg_resize = mmcv.imresize(results['gt_semantic_seg'], (self.grid_section, self.grid_section))
            seg_crop1 = mmcv.imrotate(seg_resize, degree1)
            seg_crop2 = mmcv.imrotate(seg_resize, degree2)

            # time_sticker = time.time()  # 保存相关图片
            # img_crop_save1 = mmcv.imwrite(img_crop1, "newback/imgs_segs/imgs_head/0323_imgcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop1, "newback/imgs_segs/imgs_head/0323_segcrop1" + str(degree_index1)+'_'+str(results['sunrgbd_rotation1'])+'_'+results['img_info']['ann'])
            #
            # img_crop_save1 = mmcv.imwrite(img_crop2, "newback/imgs_segs/imgs_head/0323_imgcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['filename'])
            # seg_crop_save1 = mmcv.imwrite(seg_crop2, "newback/imgs_segs/imgs_head/0323_segcrop2" + str(degree_index2)+'_'+str(results['sunrgbd_rotation2'])+'_'+results['img_info']['ann'])

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
        for i in range(img_crop_rotated1.shape[0]):
            for j in range(img_crop_rotated1.shape[1]):
                ori_img1[j + paste_location1[0][1]][i + paste_location1[0][0]] = img_crop_rotated1[j][
                    i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                ori_seg1[j + paste_location1[0][1]][i + paste_location1[0][0]] = seg_crop_rotated1[j][i]

        for i in range(img_crop_rotated2.shape[0]):
            for j in range(img_crop_rotated2.shape[1]):
                ori_img2[j + paste_location2[0][1]][i + paste_location2[0][0]] = img_crop_rotated2[j][
                    i]  # 注意这个ji顺序，否则会出现对角线反转这种情况
                ori_seg2[j + paste_location2[0][1]][i + paste_location2[0][0]] = seg_crop_rotated2[j][i]

        # time_sticker = time.time()  # 保存相关图片
        if self.save_augmented:
            ori_img1_save = mmcv.imwrite(ori_img1, "newback/imgs_segs/imgs/rectangle_ori_img1_" + "loc" + str(
                results['paste_location1']) + "_rot" + str(results['sunrgbd_rotation1']) + "_" + label_name)
            ori_img2_save = mmcv.imwrite(ori_img2, "newback/imgs_segs/imgs/rectangle_ori_img2_" + "loc" + str(
                results['paste_location2']) + "_rot" + str(results['sunrgbd_rotation2']) + "_" + label_name)

            seg_crop_save1 = mmcv.imwrite(ori_seg1,
                                          "newback/imgs_segs/segs/rectangle_ori_seg1_" + "loc" + str(
                                              results['paste_location1']) + "_rot" + str(
                                              results['sunrgbd_rotation1']) + "_" + label_name)
            seg_crop_save2 = mmcv.imwrite(ori_seg2,
                                          "newback/imgs_segs/segs/rectangle_ori_seg2_" + "loc" + str(
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
        if self.loc_specify is not None:
            location1 = self.loc_specify  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
            location2 = self.loc_specify
        else:
            location1 = random.randint(0, self.grid ** 2)  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
            location2 = random.randint(0, self.grid ** 2)
        results['paste_location1'] = location1  # 存储位置信息
        results['paste_location2'] = location2
        paste_location1 = self.grid_location[location1]
        paste_location2 = self.grid_location[location2]

        self._resize_img(results, self.resize)  # 将图片和深度进行缩放
        # gt_seg_resize = self._resize_seg_beyond255(results["gt_semantic_seg"], self.resize)  # 将depth压缩至255
        # results["gt_semantic_seg"] = gt_seg_resize  # 这个就是原图进行压缩

        '''·
            # 如果 使用正方形九宫格旋转填充
            self._global_rotate_img_and_seg(results)
            self._paste_img_and_seg(results, paste_location)
        '''
        # 如果使用圆形旋转九宫格填充
        self._rotate_img_and_seg(results, paste_location1, paste_location2)  # 传入paste_location 是为了local缩放贴回时需要的信息
        self._paste_img_and_seg(results, paste_location1, paste_location2, label_name)

        if np.isnan(results['img_aug1']).any() or np.isnan(results['img_aug2']).any():
            self.logger.info('Transforms_Img_HHA info:\n' + 'the img_aug exists problem!' + '\n')

        # ori_img2_save = mmcv.imwrite(results['img_aug1'], "newback/imgs_segs/imgs/001rectangle_ori_img2_.jpg")
        #
        # seg_crop_save1 = mmcv.imwrite(results['gt_semantic_seg'],
        #                               "newback/imgs_segs/segs/002rectangle_ori_seg1_" + ".jpg"
        #
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class RandomRotateImageHHA(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 angle,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        # if isinstance(degree, (float, int)):
        #     assert degree > 0, f'degree {degree} should be positive'
        #     self.degree = (-degree, degree)
        # else:
        self.angle = angle

        # angle 部分的初始化
        angle_section = 360 / 4
        self.angle_list = [i * angle_section for i in range(4)]
        # assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
        #                               f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        degree_index1 = random.randint(0, self.angle)  # degree_index1 和 degree_index2 可能一样
        results['sunrgbd_rotation1'] = degree_index1
        results['sunrgbd_rotation2'] = degree_index1
        degree = self.angle_list[degree_index1]
        # rotate = True if np.random.rand() < self.prob else False
        # degree = np.random.uniform(min(*self.degree), max(*self.degree))
        # degree = self.degree

        # rotate image
        img = mmcv.imrotate(
            results['img'],
            angle=degree,
            border_value=self.pal_val,
            center=self.center,
            auto_bound=self.auto_bound)

        results['img_aug1'] = img
        results['img_aug2'] = img

        # rotate segs

        gt_semantic_seg = mmcv.imrotate(
            results['gt_semantic_seg'],
            angle=degree,
            border_value=self.seg_pad_val,
            center=self.center,
            auto_bound=self.auto_bound,
            interpolation='nearest')

        results['gt_semantic_seg_aug1'] = gt_semantic_seg
        results['gt_semantic_seg_aug2'] = gt_semantic_seg
        # ori_img2_save = mmcv.imwrite(results['img'], "newback/imgs_segs/imgs/001rectangle_ori_img2_.jpg")
        #
        # seg_crop_save1 = mmcv.imwrite(results['gt_semantic_seg'],
        #                               "newback/imgs_segs/segs/002rectangle_ori_seg1_" + ".jpg"
        #                                  )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class ResizeImgHHA(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_size=256):

        self.img_size = img_size


    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = mmcv.imresize(
            results['img'], (self.img_size, self.img_size), return_scale=True)
        results['ori_img'] = img
        results['img'] = img  # 这一步下面，img相关信息开始更新
        # results['img_aug2'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""

        gt_seg = mmcv.imresize(
            results['gt_semantic_seg'], (self.img_size, self.img_size), interpolation='nearest')
        results['ori_gt_semantic_seg'] = gt_seg
        results['gt_semantic_seg'] = gt_seg
        # results['gt_semantic_seg_aug2'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str
