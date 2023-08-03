# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
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
    # torch.cuda.set_device(0)  # adding this line fixed it
    args = parse_args()
    # ck = torch.load(args.checkpoint, map_location='cuda:0')
    # type = args.type
    # output_dict = None
    #
    # for key, value in ck.items():
    #     # if key.startswith('backbone'):
    #     output_dict = value  # key[9:0]这个根据需要进行修改

    data = np.load(args.checkpoint, allow_pickle=True).item()  # 读取
    for key, value in data:
        # if key.startswith('backbone'):
        output_dict = value  # key[9:0]这个根据需要进行修改


if __name__ == '__main__':
    main()
