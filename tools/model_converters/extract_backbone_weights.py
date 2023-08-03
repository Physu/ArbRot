# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='destination file name')
    # type:
    # 1 means shapeconv
    parser.add_argument('--type', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith('.pth')
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    type = args.type
    output_dict = dict(state_dict=dict(), author='MARC')
    has_backbone = False
    if type == 1:  # for shapeconv segmentation
        for key, value in ck['state_dict'].items():
            # if key.startswith('backbone'):
            if key.startswith('encoder_q.0.'):  # bakcbone，此处改为encoder_q
                output_dict[key[12:]] = value  # key[9:0]这个根据需要进行修改
                has_backbone = True
    elif type == 2:  # 用于ShapeConv
        for key, value in ck['state_dict'].items():
            # if key.startswith('backbone'):
            if key.startswith('backbone_q.'):  # bakcbone
                output_dict['state_dict'][key[11:]] = value  # key[9:0]这个根据需要进行修改
                has_backbone = True

    elif type == 3:  # 用于mmclassification
        for key, value in ck['state_dict'].items():
            # if key.startswith('backbone'):
            if key.startswith('backbone_q.'):  # bakcbone，此处改为encoder_q
                output_dict['state_dict']['backbone.'+key[11:]] = value  # key[9:0]这个根据需要进行修改
                has_backbone = True

    elif type == 4:  # 用于mmclassification 转 mmsegmentation，主要mmclas多了个backbone.
        for key, value in ck['state_dict'].items():
            # if key.startswith('backbone'):
            if key.startswith('backbone.'):  # bakcbone，此处改为encoder_q
                output_dict['state_dict'][key[9:]] = value  # key[9:0]这个根据需要进行修改
                has_backbone = True
    elif type == 5:  # 用于mmsegmentation 转 mmclassification，主要mmseg基础上多了个backbone.
        for key, value in ck['state_dict'].items():
            # if key.startswith('backbone'):
            # bakcbone，此处改为encoder_q
            output_dict['state_dict']['backbone.'+key] = value  # key[9:0]这个根据需要进行修改
            has_backbone = True
    elif type == 6:  # 用于AdaShare backbone权重的提取
        for key, value in ck['mtl-net'].items():
            if key.startswith('backbone'):
                if key.startswith('backbone.blocks'):
                    output_dict['state_dict']['layer'+key[16:]] = value  # key[9:0]这个根据需要进行修改

                if key.startswith('backbone.ds'):
                    output_dict['state_dict']['layer' + key[12:15]+'downsample.'+key] = value  # key[9:0]这个根据需要进行修改
                has_backbone = True


    if not has_backbone:
        raise Exception('Cannot find a backbone module in the checkpoint.')
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
