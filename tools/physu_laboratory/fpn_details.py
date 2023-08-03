import torch
from mmseg.models.necks.fpn import FPN
in_channels = [2, 3, 5, 7]
scales = [340, 170, 84, 43]
inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
self = FPN(in_channels, 11, len(in_channels)).eval()
outputs = self.forward(inputs)
for i in range(len(outputs)):
    print(f'outputs[{i}].shape = {outputs[i].shape}')