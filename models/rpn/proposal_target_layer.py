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
import numpy.random as npr
from ..config import rfcn_config


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, n_classes):
        super(_ProposalTargetLayer, self).__init__()
        self.num_classes = n_classes

    def forward(self, all_rois, gt_boxes):

        gt_boxes = torch.from_numpy(gt_boxes).float()

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes], 1)

        if rfcn_config.OHEM:
            rois_per_image = 300
        else:
            rois_per_image = rfcn_config.ROI_BATCH_SIZE

        fg_rois_per_image = int(np.round(rfcn_config.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # Generate examples using OHEM
        

        return all_rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

