import torch.nn as nn

# from ..utils import accuracy
from ..registry import HEADS
from ..utils import build_norm_layer  #, MultiPooling


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


class MultiPooling(nn.Module):
    """Pooling layers for features from multiple depth."""

    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=6, stride=1, padding=0)
        ]
    }
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 2]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 8192]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][l])
                for l in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][l])
                for l in in_indices
            ])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return [p(xx) for p, xx in zip(self.pools, x)]


@HEADS.register_module()
class MultiClsHead(nn.Module):
    """Multiple classifier heads.
    """

    FEAT_CHANNELS = {'resnet50': [64, 256, 512, 1024, 2048]}
    FEAT_LAST_UNPOOL = {'resnet50': 2048 * 7 * 7}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 with_last_layer_unpool=False,
                 backbone='resnet50',
                 norm_cfg=dict(type='BN'),
                 num_classes=1000):
        super(MultiClsHead, self).__init__()
        assert norm_cfg['type'] in ['BN', 'SyncBN', 'GN', 'null']

        self.with_last_layer_unpool = with_last_layer_unpool
        self.with_norm = norm_cfg['type'] != 'null'

        self.criterion = nn.CrossEntropyLoss()

        self.multi_pooling = MultiPooling(pool_type, in_indices, backbone)

        if self.with_norm:
            self.norms = nn.ModuleList([
                build_norm_layer(norm_cfg, self.FEAT_CHANNELS[backbone][l])[1]
                for l in in_indices
            ])

        self.fcs = nn.ModuleList([
            nn.Linear(self.multi_pooling.POOL_DIMS[backbone][l], num_classes)
            for l in in_indices
        ])
        if with_last_layer_unpool:
            self.fcs.append(
                nn.Linear(self.FEAT_LAST_UNPOOL[backbone], num_classes))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        if self.with_last_layer_unpool:
            last_x = x[-1]
        x = self.multi_pooling(x)
        if self.with_norm:
            x = [n(xx) for n, xx in zip(self.norms, x)]
        if self.with_last_layer_unpool:
            x.append(last_x)
        x = [xx.view(xx.size(0), -1) for xx in x]
        x = [fc(xx) for fc, xx in zip(self.fcs, x)]
        return x

    def loss(self, cls_score, labels):
        losses = dict()
        for i, s in enumerate(cls_score):
            # keys must contain "loss"
            losses['loss.{}'.format(i + 1)] = self.criterion(s, labels)
            losses['acc.{}'.format(i + 1)] = accuracy(s, labels)
        return losses
