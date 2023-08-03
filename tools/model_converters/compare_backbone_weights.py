# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from scipy.stats import pearsonr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint1', type=str, help='checkpoint file')
    parser.add_argument('checkpoint2', type=str, help='checkpoint file')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ck1 = torch.load(args.checkpoint1, map_location=torch.device('cpu'))
    ck2 = torch.load(args.checkpoint2, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author='MMSelfSup')
    has_backbone = False
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    loss_similarity = torch.tensor(0.0, dtype=torch.float32).cuda()
    loss_pearson = 0.0
    iter = torch.tensor(0.0, dtype=torch.float32).cuda()
    for value1, value2 in zip(ck1['state_dict'].items(), ck2['state_dict'].items()):
        if not value1[0].endswith('num_batches_tracked'):
            loss_similarity = loss_similarity + similarity(value1[1], value2[1])
            # loss_pearsonr = loss_pearsonr + pearsonr(value1[1].cpu().numpy(), value2[1].cpu().numpy())
            x1 = value1[1].flatten().unsqueeze(0)
            x2 = value2[1].flatten().unsqueeze(0)
            vx = x1 - torch.mean(x1)
            vy = x2 - torch.mean(x2)
            cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            loss_pearson = loss_pearson + cost

            iter = iter + 1
    losses = loss_similarity / (iter + 1e-8)
    losses2 = loss_pearson / (iter+1e-8)
    print(f"iters:{iter}, cosine_similarity:{losses} pearsonr:{losses2}")


def dot_product(v1, v2):
        """Get the dot product of the two vectors.
        if A = [a1, a2, a3] && B = [b1, b2, b3]; then
        dot_product(A, B) == (a1 * b1) + (a2 * b2) + (a3 * b3)
        true
        Input vectors must be the same length.
        """
        return torch.sum(torch.mul(v1, v2))

def magnitude(vector):
        """Returns the numerical length / magnitude of the vector."""
        return torch.sqrt(dot_product(vector, vector))

def similarity(v1, v2):
        """Ratio of the dot product & the product of the magnitudes of vectors."""
        return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)




if __name__ == '__main__':
    main()
