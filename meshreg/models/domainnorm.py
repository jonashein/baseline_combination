# Source: Domain-invariant Stereo Matching Networks by Zhang et. al. (ECCV 2020)
# Paper: https://arxiv.org/pdf/1911.13287.pdf
# Code: https://github.com/feihuzhang/DSMNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainNorm(nn.Module):

    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1,channel,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            return F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias