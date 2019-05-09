import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from ..config import rfcn_config as cfg
from torch.autograd import Variable
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorLayer


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
        self.nc_score_out = len(anchor_dimensions)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(anchor_dimensions) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_dimensions, scale)

        # defined anchor target layer
        self.RPN_anchor_target = _AnchorLayer()

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

    def forward(self, base_features, image_metadata, gt_boxes):
        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_features), inplace=True)

        # get rpn classification score
        rpn_classification_score = self.RPN_cls_score(rpn_conv1)

        rpn_classification_score_reshape = self.reshape(rpn_classification_score, 2)
        rpn_classification_prob_reshape = F.softmax(rpn_classification_score_reshape, dim=1)
        rpn_classification_prob = self.reshape(rpn_classification_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_predictions = self.RPN_bbox_pred(rpn_conv1)

        rois = self.RPN_proposal(rpn_classification_prob.data, rpn_bbox_predictions.data, image_metadata)

        # print("ROIS AND SHAPE:", rois, rois.shape)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # Get anchor targets
            labels, targets = self.RPN_anchor_target(rpn_classification_prob.data, gt_boxes, image_metadata)

            # Unsqueezing to match dimensions
            rpn_classification_prob = torch.unsqueeze(rpn_classification_prob, 4)

            # Compute cross-entropy classification loss
            valid_indices = labels.view(-1).ne(-1).nonzero()
            pred = rpn_classification_prob.view(-1)[valid_indices].squeeze()
            actual = labels.view(-1)[valid_indices].float().squeeze()

            # self.rpn_loss_cls = F.cross_entropy(pred, actual)
            self.rpn_loss_cls = F.binary_cross_entropy(pred, actual.cuda())

            # Compute smooth l1 bbox regression loss
            self.rpn_loss_box = self.smooth_l1_loss(rpn_bbox_predictions, targets.cuda(), labels,
                                                    delta=cfg.RPN_L1_DELTA, dim=[1, 2, 3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box

    def smooth_l1_loss(self, bb_prediction, bb_target, bb_labels, delta=1.0, dim=[1]):
        """
        Loss function for Smooth L1 taken from Wikipedia (https://en.wikipedia.org/wiki/Huber_loss) and
            reference code.

        :param bb_prediction:
        :param bb_target:
        :param bb_labels:
        :param delta:
        :param dim:
        :return:
        """

        delta_sq = delta ** 2

        bb_prediction = bb_prediction.view(bb_prediction.shape[0], bb_prediction.shape[1] // 4,
                                           bb_prediction.shape[2], bb_prediction.shape[3], 4)

        difference = torch.abs(bb_prediction - bb_target).float()

        mask = (bb_labels == 1.0).float()

        # print("LABEL:", bb_labels.shape, bb_labels.ge(0).float().sum())
        # print("MASK:", mask.shape, mask.sum())
        """
        Mask to take only positive anchors into consideration
        """

        weight = 1.0
        """
        Normalizing factor
        """

        if cfg.UNIFORM_EXAMPLE_WEIGHTING:
            weight = 1.0 / torch.sum(bb_labels >= 0).float()

        l1_apply_mask = difference <= delta

        # print("L1 MASK", l1_apply_mask)

        l1_apply_mask = l1_apply_mask.float()

        diff_sq = torch.pow(difference, 2)

        # print("DIFF_SQ", diff_sq)

        LHS = (l1_apply_mask * (diff_sq * 0.5))

        # print("LHS:", LHS)

        RHS = ((1 - l1_apply_mask) * ((delta * difference) - (0.5 * delta_sq)))

        # print("RHS:", RHS)

        losses = LHS + RHS

        losses = (losses * weight.cuda())

        # print("Losses BEFORE mask:", losses)

        losses = losses * mask.cuda()
        # print("Losses after mask:", losses)
        for i in sorted(dim, reverse=True):
            losses = losses.sum(i)
        losses = losses.mean()

        # print("LOSSES AFTER MEAN", losses)

        return losses
