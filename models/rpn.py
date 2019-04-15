from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import pdb
import time

from proposal_layer import _ProposalLayer


# Referred from https://github.com/jmerkow/R-FCN.rsna2018
class RPN(nn.Module):
    """ Region proposal network """
    def __init__(self, input_channels, anchor_dimensions, stride, scale):
        super(RPN, self).__init__()
        
        self.input_channels = input_channels  # get depth of input feature map, e.g., 512
        self.anchor_dimensions = anchor_dimensions
        self.feat_stride = stride

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.input_channels, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(anchor_dimensions) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(anchor_dimensions) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_dimensions, scale)

        # TODO: define anchor target layer

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal(base_feat, gt_boxes, im_info)


        return rois, self.rpn_loss_cls, self.rpn_loss_box
