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

from ..utils import anchors
from ..utils import image_plotting
from ..utils import image_processing
from ..config import rpn_config

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

    def forward(self, scores, bbox_deltas, image_info):

        # Algorithm
        # 1. At each location i (h,w) generate A anchors of different scales and ratios
        # 2. Apply predicted bounding box deltas to each of the A anchors
        # 3. Crop adjusted bounding boxes so that they are within the feature dimensions
        # 4. Remove those predicted bounding boxes that have a size lesser than some threshold
        # 5. Associate each bounding box with the scores predicted
        # 6. Sort all (bounding box, score) pairs by score from highest to lowest
        # 7. Take top pre_nms_topN proposals before NMS
        # 8. Apply NMS with threshold 0.7 to remaining proposals
        # 9. Take after_nms_topN proposals after NMS
        # 10. Return the top proposals (-> RoIs top, scores top)

        print(scores)

        return []

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
