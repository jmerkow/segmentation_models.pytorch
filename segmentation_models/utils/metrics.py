# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import torch.nn as nn
from . import functions as F


class IoUMetric(nn.Module):

    __name__ = 'iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return F.iou(y_pr, y_gt, self.eps, self.threshold, self.activation)


class FscoreMetric(nn.Module):

    __name__ = 'f-score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold
        self.beta = beta

    def forward(self, y_pr, y_gt):
        return F.f_score(y_pr, y_gt, self.beta, self.eps, self.threshold, self.activation)


class DICEMetric(IoUMetric):

    __name__ = 'dice'

    def forward(self, y_pr, y_gt):
        IoU = super(DICEMetric, self). forward(y_pr, y_gt)
        return (2. * IoU) / (1. + IoU)
