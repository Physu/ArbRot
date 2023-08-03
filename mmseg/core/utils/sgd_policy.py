import torch
from torch.optim import SGD
from mmcv.runner.optimizer.builder import OPTIMIZERS

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


@OPTIMIZERS.register_module()
class SGDPolicy(SGD):
    r""" 重新实现这个方法，用来更新权重的时候，不同的任务，更新不同的层
    Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, policys=None):
        super(SGDPolicy, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                 weight_decay=weight_decay, nesterov=nesterov)

        self.policys = policys

        self.layer_dict = {
            'ResNet101': [3, 4, 23, 3],
            'ResNet50': ['layer1.0.', 'layer1.1.', 'layer1.2.',
                         'layer2.0.', 'layer2.1.', 'layer2.2.', 'layer2.3.',
                         'layer3.0.', 'layer3.1.', 'layer3.2.', 'layer3.3.', 'layer3.4.', 'layer3.5.',
                         'layer4.0.', 'layer4.1.', 'layer4.2.'],  # [3, 4, 6, 3],
            'ResNet34': ['layer1.0.', 'layer1.1.', 'layer1.2.',
                         'layer2.0.', 'layer2.1.', 'layer2.2.', 'layer2.3.',
                         'layer3.0.', 'layer3.1.', 'layer3.2.', 'layer3.3.','layer3.4.', 'layer3.5.',
                         'layer4.0.', 'layer4.1.', 'layer4.2.'],  # [3, 4, 6, 3],
            'ResNet18': [2, 2, 2, 2]
        }

    def __setstate__(self, state):
        super(SGDPolicy, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, grad_dict, named_params_dict, policys=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for policy, key in zip(policys, grad_dict):
            # 获取对应的keys和values
                for p, name in zip(group['params'], named_params_dict):
                    if p.grad is None:
                        continue

                    if name.startswith('module.backbone_q.layer'):
                        name_list = name.split('.')
                        layer_name = name_list[2] + '.' + name_list[3] + '.'
                        layer_index = self.layer_dict['ResNet50'].index(layer_name)
                        # print(f'name:{name[18:27]} and index:{index}')

                        if name in grad_dict[key]:
                            # d_p = p.grad
                            d_p = grad_dict[key][name] * policy[layer_index][0]  # 第一位是对应的权重
                            if weight_decay != 0:
                                d_p = d_p.add(p, alpha=weight_decay)
                            if momentum != 0:
                                param_state = self.state[p]
                                if 'momentum_buffer' not in param_state:
                                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                                else:
                                    buf = param_state['momentum_buffer']
                                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                                if nesterov:
                                    d_p = d_p.add(buf, alpha=momentum)
                                else:
                                    d_p = buf
                            p.add_(d_p, alpha=-group['lr'])

                    else:
                        d_p = p.grad   # 共执行五次，每次是对应代理任务产生的梯度
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'momentum_buffer' not in param_state:
                                buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = param_state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        p.add_(d_p, alpha=-group['lr'])

        return loss
