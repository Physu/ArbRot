import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
import torch


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


@HEADS.register_module()
class RotHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    注意这里，其实实现的是RotNet，最多四个角度
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=4,
                 rot_weight=1.):
        super(RotHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.rot_weight = rot_weight

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[3]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def generate_bce_weight(self, x):
        '''
        用来生成所需要的weight信息，因为要预测360°的结果，所以设计有正负2度的容忍度
        :param x:
        :param mu: mean value
        :param sigma: std value
        :return:
        '''
        pdf = 1 - 1 / ((torch.exp(-x)/2)+(torch.exp(x)/2))
        return pdf

    def loss(self, cls_score, labels):
        '''
        import torch
        a = torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
        b = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]])
        tar = torch.tensor([4])
        l11 = self.criterion(a, tar)
        l22 = self.criterion(b, tar)
        注意这里，l11 和 l22 的数值是一样的，这个对于预测角度来说肯定是不合理，对于a，这种预测结果，应该
        要比b这种预测结果的loss 更小一些，而不是二者的loss相同
        :param cls_score:
        :param labels:
        :return:
        '''

        # cls_score_max_index = torch.max(cls_score[0], dim=1)[1]  # 返回值有两个，一个最大值，一个对应的序列索引
        # gap = cls_score_max_index - labels  # 预测和实际之间的差别
        # rot_bce_weight = self.generate_bce_weight(gap.to(torch.float32))

        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss_rot'] = self.criterion(cls_score[0], labels) * self.rot_weight

        return losses
