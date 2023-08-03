from scipy.stats import beta
import numpy as np

# a = input()  # 贝塔分布的alpha值
# b = input()  # 贝塔分布的beta值
a = 0.8
b = 0.8
x = np.arange(0.01, 1, 0.01)  # 给定的输入数据
print(beta.pdf(x, a, b))

import torchvision.transforms.functional as F

from PIL import Image

# 图片路径，相对路径
image_path = "./tools/physu_laboratory/test.jpg"
# 读取图片
image = Image.open(image_path)
# 输出图片维度
print("image_shape: ", image.size)
# 显示图片
# image.show()
i = 100
j = 100
h = 100
w = 100
size = 300
img = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
img.save('./tools/physu_laboratory/test_resized.jpg')
