# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    # parser.add_argument('output', type=str, help='destination file name')
    # type:
    # 1 means shapeconv
    parser.add_argument('--type', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu')
    type = args.type
    output_dict = None

    for key, value in ck.items():
        # if key.startswith('backbone'):
        output_dict = value  # key[9:0]这个根据需要进行修改

if __name__ == '__main__':
    main()
