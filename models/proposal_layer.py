from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import yaml
import pdb
import torchvision

from .utils import anchors
from .utils import image_plotting
from .utils import image_processing
from .config import rpn_config

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, box_sizes, scale):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._box_sizes = box_sizes
        self._scale = scale

    def forward(self, features, b_boxes, im_path, image):

        orig_b_boxes = b_boxes

        # Convert bounding boxes to a scale of the feature map from the input image
        b_boxes = image_processing.scale_boxes(b_boxes, self._scale, 'down')
        
        # Generate anchors
        positive_anchors, negative_anchors = anchors.generate_anchors(features, 100, b_boxes, self._box_sizes)

        # Convert anchors to original image size
        b_boxes = image_processing.scale_boxes(b_boxes, self._scale, 'up')
        positive_anchors = image_processing.scale_boxes(positive_anchors, self._scale, 'up')
        negative_anchors = image_processing.scale_boxes(negative_anchors, self._scale, 'up')


        # Plot anchors on original image
        image_plotting.plot_boxes(image[0], positive_anchors, negative_anchors, b_boxes)

        return []

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
