import torch

x1 = torch.randn(2,3,5,5)
x2 = torch.randn(2,3,5,5)
conv1 = torch.nn.Conv2d(3,3,(2,2))
conv2 = torch.nn.Conv2d(3,3,(2,2))
conv3 = torch.nn.Conv2d(3,3,(2,2))
x1_conv1 = conv1(x1)
x2_conv1 = conv1(x2)
x1_conv2 = conv2(x1_conv1)
x2_conv2 = conv2(x2_conv1)
x1_conv3 = conv3(x1_conv2)
x2_conv3 = conv3(x2_conv2)
res3_conv1 = conv1(torch.cat((x1, x2), dim=0))
res3_conv2 = conv2(res3_conv1)
res3_conv3 = conv3(res3_conv2)

    # shape = (2, 8, 6, 1)