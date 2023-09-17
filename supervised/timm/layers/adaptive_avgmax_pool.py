""" PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim

Both a functional and a nn.Module version of the pooling is provided.

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_pool2d import AttentionPool2d
from .median_pool import MedianPool2d
from .cbam import CbamModule
from .other_pool import LSEPool, GEM, GeneralizedMP, OTKPool, GEPooling, HOWPooling, CAPooling, ViTPooling, SlotPooling

#TODO: Fix this!
import sys
sys.path.append("..")
from sp import SimPool

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type
    return x


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3), keepdim=not self.flatten)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, dim, output_size=1, pool_type='fast', gamma=2.0, flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == 'attnpool':
            self.pool = AttentionPool2d(in_features=dim, feat_size=(7, 7), out_features=512)
        elif pool_type == 'cbam':
            self.pool = CbamModule(channels=dim, spatial_kernel_size=7)
        elif pool_type == 'median':
            self.pool = MedianPool2d(kernel_size=7)
        elif pool_type == 'learnlse':
            self.pool = LSEPool(r=10, learnable=True)
        elif pool_type == 'lse':
            self.pool = LSEPool(r=1, learnable=False)
        elif pool_type == 'gem':
            self.pool = GEM(gamma=gamma, kernel=(7, 7))
        elif pool_type == 'genmp':
            self.pool = GeneralizedMP(lamb=1e3)
        elif pool_type == 'otk':
            self.pool = OTKernel(in_dim=dim, out_size=1, heads=1, eps=25, max_iter=1)
        elif pool_type == 'se':
            self.pool = GEPooling(channels=dim, extra_params=False, extent=0, output_size=output_size)
        elif pool_type == 'ge':
            self.pool = GEPooling(channels=dim, extra_params=True, extent=2, output_size=output_size)
        elif pool_type == 'how':
            self.pool = HOWPooling(input_dim=dim, dim_reduction=512, kernel_size=3)
        elif pool_type == 'vit':
            self.pool = ViTPooling(d_model=dim, num_heads=4)
        elif pool_type == 'cait':
            self.pool = CAPooling(embed_dim=dim, num_heads=4, iterations=1)
        elif pool_type == 'slot':
            self.pool = SlotPooling(dim=dim, num_slots=3)
        elif pool_type == 'simpool':
            self.pool = SimPool(dim=dim, gamma=gamma)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

