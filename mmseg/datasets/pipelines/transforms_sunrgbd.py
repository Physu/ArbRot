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


# # register all existing transforms in torchvision
# _EXCLUDED_TRANSFORMS = ['GaussianBlur']
# for m in inspect.getmembers(_transforms, inspect.isclass):
#     if m[0] not in _EXCLUDED_TRANSFORMS:
#         PIPELINES.register_module(m[1])


@PIPELINES.register_module()
class ResizeRotatePaste(object):
    """First resize, then rotate, finally Paste to the original image & seg.

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
                 resize,
                 small_resize):
        self.num = 0
        self.resize = resize
        self.small_resize = small_resize

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            results['img'], self.resize, return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img  # 这一步下面，img相关信息开始更新
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor  # 调整的scale信息

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            gt_seg = mmcv.imresize(
                results[key], self.resize)
            results[key] = gt_seg

    def _rotate_img_and_seg(self, results):
        degrees = [0., 90., 180., 270.]
        degree_index = random.randint(0, 3)
        degree = degrees[degree_index]

        # rotate image
        results['img_rotate'] = mmcv.imrotate(
            results['img'],
            angle=degree, )

        # rotate segs
        for key in results.get('seg_fields', []):
            results[key + '_rotate'] = mmcv.imrotate(
                results[key],
                angle=degree)

            results['sunrgbd_rotation'] = degree_index  # 将旋转角度索引存入

    def _paste_img_and_seg(self, results):

        img_small = mmcv.imresize(results['img_rotate'], self.small_resize)
        seg_small = mmcv.imresize(results['gt_semantic_seg_rotate'], self.small_resize)

        background = results['img']
        seg_background = results['gt_semantic_seg']
        '''
            —— —— —— —— w
            |
            |
            |
            |
            h
        '''
        locations = [[[0, 0], [75, 75]], [[0, 75], [75, 150]], [[0, 150], [75, 225]],
                     [[75, 0], [150, 75]], [[75, 75], [150, 150]], [[75, 150], [150, 225]],
                     [[150, 0], [225, 75]], [[150, 75], [225, 150]], [[150, 150], [225, 225]]]
        location = random.randint(0, 8)
        results['sunrgbd_location'] = location

        paste_location = locations[location]
        for w in range(paste_location[0][0], paste_location[1][0]):
            for h in range(paste_location[0][1], paste_location[1][1]):
                # 三通道
                background[:, :, 0][w][h] = img_small[:, :, 0][w - paste_location[0][0]][h - paste_location[0][1]]
                background[:, :, 1][w][h] = img_small[:, :, 1][w - paste_location[0][0]][h - paste_location[0][1]]
                background[:, :, 2][w][h] = img_small[:, :, 2][w - paste_location[0][0]][h - paste_location[0][1]]
                seg_background[w, h] = seg_small[w - paste_location[0][0]][h - paste_location[0][1]]

        results['img'] = background  # 处理之后的图片
        results['gt_semantic_seg'] = seg_background  # 处理之后的depth 图
        time_sticker = time.time()  # 保存相关图片
        if self.num < 5:
            self.num = self.num + 1
            background_ret = mmcv.imwrite(background, "newback/images/background" + str(int(time_sticker)) + ".jpg")
            seg_background = mmcv.imwrite((seg_background * 10000).astype(np.uint8),
                                          "newback/images/seg_background" + str(int(time_sticker)) + ".png")
            # seg_background = mmcv.imwrite(seg_background, "newback/images/seg_background"+str(int(time_sticker))+".png")

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        self._resize_img(results)
        self._resize_seg(results)
        self._rotate_img_and_seg(results)
        self._paste_img_and_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(image width and height={self.resize}, ' \
                    f'small image width and height={self.degree})'
        return repr_str


@PIPELINES.register_module()
class RotateCircle(object):
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
        #                              "newback/imgs_segs/segs/seg_crop1_" +   + ".jpg")
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
            ori_img1_save = mmcv.imwrite(ori_img1, "newback/imgs_segs/imgs/ori_img1_" + "loc"+str(results['paste_location1']) + "_rot"+str(results['sunrgbd_rotation1']) + "_" + label_name)
            ori_img2_save = mmcv.imwrite(ori_img2, "newback/imgs_segs/imgs/ori_img2_" + "loc"+str(results['paste_location2']) + "_rot"+str(results['sunrgbd_rotation2']) + "_" + label_name)

            seg_crop_save1 = mmcv.imwrite((ori_seg1 * 10000).astype(np.uint8),
                                          "newback/imgs_segs/segs/ori_seg1_" + "loc"+str(results['paste_location1']) + "_rot"+str(results['sunrgbd_rotation1']) + "_" + label_name)
            seg_crop_save2 = mmcv.imwrite((ori_seg2 * 10000).astype(np.uint8),
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
        location1 = random.randint(0, self.grid ** 2)  # random.randint(m,n) 返回m到n之间的随机整数，但不包括n
        location2 = random.randint(0, self.grid ** 2)
        results['paste_location1'] = location1  # 存储位置信息
        results['paste_location2'] = location2
        paste_location1 = self.grid_location[location1]
        paste_location2 = self.grid_location[location2]

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


########################################################### borrow from openselfsup, z
# custom transforms
@PIPELINES.register_module()
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
            torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
            torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, results):
        img = results['img']
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        results['img'] = img.add(rgb.view(3, 1, 1).expand_as(img))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max, prepare_for_moco=False):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prepare_for_moco = prepare_for_moco

    def __call__(self, results):
        fn_idx = random.randint(0, 1)
        if self.prepare_for_moco:
            if fn_idx == 0:
                img_aug1 = results['img_aug1']
                img_aug2 = results['img_aug2']
                sigma1 = np.random.uniform(self.sigma_min, self.sigma_max)
                sigma2 = np.random.uniform(self.sigma_min, self.sigma_max)
                img_aug1 = Image.fromarray(img_aug1)  # ndarray 转换成为 PIL.Image 反之则numpy.array(img)：img对象转化为np数组
                img_aug2 = Image.fromarray(img_aug2)
                img_aug1 = img_aug1.filter(ImageFilter.GaussianBlur(radius=sigma1))
                img_aug2 = img_aug2.filter(ImageFilter.GaussianBlur(radius=sigma2))
                img_aug1 = np.array(img_aug1)
                img_aug2 = np.array(img_aug2)
                results['img_aug1'] = img_aug1
                results['img_aug2'] = img_aug2
        else:
            if fn_idx == 0:
                img = results['img']
                sigma = np.random.uniform(self.sigma_min, self.sigma_max)
                img = Image.fromarray(img)  # ndarray 转换成为 PIL.Image 反之则numpy.array(img)：img对象转化为np数组
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                img = np.array(img)
                results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733.
    solarization: an optional color transformation x → x·1{x<0.5} + (1−x)·1{x≥0.5} for pixels with values in [0, 1].
    """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, results):
        img = results['img']
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 - img)  # 从这里看，图片没经过归一化处理
        results['img'] = Image.fromarray(img.astype(np.uint8))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ColorJitter(object):
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

    def __init__(self, brightness=0, contrast=0, saturation=0, prepare_for_moco=False):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prepare_for_moco = prepare_for_moco

    def __call__(self, results):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        if self.prepare_for_moco:
            brightness_factor = random.uniform(0, self.brightness)
            contrast_factor = random.uniform(0, self.contrast)
            saturation_factor = random.uniform(0, self.saturation)
            img_aug1 = results['img_aug1']
            img_aug2 = results['img_aug2']
            fn_idx1 = torch.randperm(3)
            fn_idx2 = torch.randperm(3)# 四种图片处理方式，随机采用其中一种, 再加上不处理保持原样
            fn_idx = [fn_idx1, fn_idx2]
            img_aug = [img_aug1, img_aug2]
            img_name = ['img_aug1', 'img_aug2']
            for i in range(2):
                for fn_id in fn_idx[i]:
                    if fn_id == 0 and brightness_factor is not None:
                        img_aug[i] = adjust_brightness(img_aug[i], brightness_factor)
                        # img_aug2 = adjust_brightness(img_aug2, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img_aug[i] = adjust_contrast(img_aug[i], contrast_factor)
                        # img_aug2 = adjust_contrast(img_aug2, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img_aug[i] = adjust_color(img_aug[i], saturation_factor)
                        # img_aug2 = adjust_color(img_aug2, saturation_factor)
                results[img_name[i]] = img_aug[i]
                # results['img_aug2'] = img_aug2

        else:
            img = results['img']
            fn_idx = torch.randperm(3)  # 四种图片处理方式，随机采用其中一种, 再加上不处理保持原样
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness_factor is not None:
                    img = adjust_brightness(img, self.brightness_factor)
                elif fn_id == 1 and self.contrast_factor is not None:
                    img = adjust_contrast(img, self.contrast_factor)
                elif fn_id == 2 and self.saturation_factor is not None:
                    img = adjust_color(img, self.saturation_factor)

            results['img'] = img

        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


@PIPELINES.register_module()
class Normalize_img_and_label(object):
    """Normalize the image and label

    Added key is "img_norm_cfg and label_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, img_norm_cfg, label_norm_cfg):
        # self.mean = np.array(mean, dtype=np.float32)
        # self.std = np.array(std, dtype=np.float32)
        # self.to_rgb = to_rgb
        self.img_mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.img_std = np.array(img_norm_cfg['std'], dtype=np.float32)
        self.to_rgb = img_norm_cfg['to_rgb']
        self.label_mean = np.array(label_norm_cfg['mean'], dtype=np.float32)
        self.label_std = np.array(label_norm_cfg['std'], dtype=np.float32)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img_aug1 = results['img_aug1']
        img_aug2 = results['img_aug2']
        gt_semantic_seg_aug1 = results['gt_semantic_seg_aug1']
        gt_semantic_seg_aug2 = results['gt_semantic_seg_aug2']

        img_aug1_for_loss = results['img_aug1_for_loss']
        img_aug2_for_loss = results['img_aug2_for_loss']
        gt_semantic_seg_aug1_for_loss = results['gt_semantic_seg_aug1_for_loss']
        gt_semantic_seg_aug2_for_loss = results['gt_semantic_seg_aug2_for_loss']

        # ori_img1_save = mmcv.imwrite(img_aug1,
        #                              "newback/imgs_segs/imgs_beforenorm/img1_" + results['img_info']['filename'])
        # # ori_img2_save = mmcv.imwrite(img_aug2,
        # #                              "newback/imgs_segs/imgs_beforenorm/img2_" + results['img_info']['filename'])
        # if len(gt_semantic_seg_aug1.astype(np.uint8).shape) == 2 or gt_semantic_seg_aug1.astype(np.uint8).shape[-1] == 1:
        #     img = cv2.cvtColor(gt_semantic_seg_aug1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        #     ori_depth_save = mmcv.imwrite(img,
        #                                   "newback/imgs_segs/imgs_beforenorm/depth1_" + results['ann_info']['seg_map'])
        # ori_depth_save = mmcv.imwrite(gt_semantic_seg_aug1.astype(np.uint8),
        #                              "newback/imgs_segs/imgs_beforenorm/depth1_" + results['ann_info']['seg_map'])
        results['img_aug1_normalization'] = mmcv.imnormalize(img_aug1, self.img_mean, self.img_std, self.to_rgb)
        results['img_aug2_normalization'] = mmcv.imnormalize(img_aug2, self.img_mean, self.img_std, self.to_rgb)
        results['gt_semantic_seg_aug1_normalization'] = mmcv.imnormalize(gt_semantic_seg_aug1, self.label_mean, self.label_std)
        results['gt_semantic_seg_aug2_normalization'] = mmcv.imnormalize(gt_semantic_seg_aug2, self.label_mean, self.label_std)

        results['img_aug1_for_loss_normalization'] = mmcv.imnormalize(img_aug1_for_loss, self.img_mean, self.img_std, self.to_rgb)
        results['img_aug2_for_loss_normalization'] = mmcv.imnormalize(img_aug2_for_loss, self.img_mean, self.img_std, self.to_rgb)
        results['gt_semantic_seg_aug1_for_loss_normalization'] = mmcv.imnormalize(gt_semantic_seg_aug1_for_loss, self.label_mean,
                                                                         self.label_std)
        results['gt_semantic_seg_aug2_for_loss_normalization'] = mmcv.imnormalize(gt_semantic_seg_aug2_for_loss, self.label_mean,
                                                                         self.label_std)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str
