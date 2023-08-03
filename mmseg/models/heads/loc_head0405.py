import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

# from ..utils import accuracy # 在openselfsup 中是这样
from ..registry import HEADS
from mmcv.runner import BaseModule


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
class LocHead0405(BaseModule):
    """Simplest classifier head, with only one fc layer.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=9,
                 loss_total_weight=1.0):
        super(LocHead0405, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.loss_total_weight = loss_total_weight  # loss权重设置

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x[3])
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        '''
         The `input` is expected to contain raw, unnormalized scores for each class.
        '''
        losses['loss_loc'] = self.criterion(cls_score[0], labels) * self.loss_total_weight
        # losses['acc_loc'] = accuracy(cls_score[0], labels, topk=(1))  # 返回top1，top5
        # if len(losses['acc_loc']) == 2:
        #     losses['top1'] = losses['acc_loc'][0]
        #     losses['top5'] = losses['acc_loc'][1]
        #     del losses['acc_loc']
        return losses
