# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith('.pth')
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author='MMSelfSup')
    has_backbone = False
    for key, value in ck['state_dict'].items():
        # if key.startswith('backbone'):
        print(key+'\n')
        # if key.startswith('encoder_q.0.'):  # bakcbone，此处改为encoder_q
        #     output_dict['state_dict'][key[12:]] = value  # key[9:0]这个根据需要进行修改
        #     has_backbone = True
    if not has_backbone:
        raise Exception('Cannot find a backbone module in the checkpoint.')
    # torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
