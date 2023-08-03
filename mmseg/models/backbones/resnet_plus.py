import torch
from .resnet import ResNet
from ..builder import BACKBONES


@BACKBONES.register_module()
class ResNetAdaShare3(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.

    最开始7*7输入通道为3
    """

    def __init__(self, **kwargs):
        super(ResNetAdaShare3, self).__init__(
            deep_stem=False, avg_down=False, **kwargs)

    def forward(self, x, policy=None):
        """Forward function."""
        if policy is None:  # 没有policy和y的时候，就是原版resnet
            x_outs = self.forward_without_policy(x)
        else:
            x_outs = self.forward_policy(x, policy)

        return x_outs

    def forward_without_policy(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def forward_policy(self, x, policy):
        t = 0
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            for j in range(len(res_layer)):
                if j == 0 and res_layer[j].downsample is not None:
                    residual = res_layer[j].downsample(x)
                else:
                    residual = x
                x = res_layer[j](x)

                if policy.ndimension() == 2:  # 三个任务的时候，policy.ndimension()为2
                    # x = fx * policy[t, 0] + residual * policy[t, 1]  # 这说明对应两个不同的权重
                    x = x * policy[t, 0] + residual * policy[t, 1]  # policy用来确定是用
                else:
                    raise ValueError('policy ndimension() != %f is incorrect' % policy.ndimension())
                t += 1

            # x = res_layer(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)