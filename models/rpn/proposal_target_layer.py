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
from ..utils import anchors as anchor_ops
from ..utils.image_plotting import plot_boxes


BBOX_INSIDE_WEIGHTS = torch.FloatTensor(rfcn_config.BBOX_INSIDE_WEIGHTS)

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, n_classes):
        super(_ProposalTargetLayer, self).__init__()
        self.num_classes = n_classes
        
    def forward(self, rois, gt_boxes, features):

        batch_size = rois.shape[0]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([rois, gt_boxes], 1)
        
        if rfcn_config.verbose:
            print("\n----Proposal Target Layer----\n\n")
            print("ROIs generated:", rois.shape)
            print("Total ROIs with GT boxes:", all_rois.shape)

        rois_per_image = rfcn_config.ROI_BATCH_SIZE

        fg_rois_per_image = int(np.round(rfcn_config.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # Generate examples based on Regions generated for the upcoming PSROI layers
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, self.num_classes, features)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, features):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """

        # calculate all combinations of overlaps (rois x gt_boxes)
        overlaps = bbox_overlaps_vectorized(all_rois, gt_boxes)

        # Get max overlaps across the different GT boxes
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)

        # TODO: Check if these useless two lines make sense for a batch atleast 
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # Face label with value as 1, in Face R-FCN there are only two classes - Face(1), No-Face(0)
        face_label = 1
        background_label = 0

        labels_batch = all_rois.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 4).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 4).zero_()

        # Guard against the case when an image has fewer than max_fg_rois_per_image foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= rfcn_config.FACE_THRESH).view(-1)

            num_fg_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < rfcn_config.BG_THRESH_HI) &
                                    (max_overlaps[i] >= rfcn_config.BG_THRESH_LO)).view(-1)

            bg_num_rois = bg_inds.numel()

            if rfcn_config.verbose:
                print("\nFace and BG ROIs:", num_fg_rois, bg_num_rois)

            if num_fg_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, num_fg_rois)
                rand_num = torch.from_numpy(np.random.permutation(num_fg_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                rand_num = torch.floor(torch.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = rand_num.type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif num_fg_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = torch.floor(torch.rand(rois_per_image) * num_fg_rois)
                rand_num = rand_num.type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]

                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and num_fg_rois == 0:
                # sampling bg
                rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois)
                rand_num = rand_num.type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and num_fg_rois = 0, this should not happen!")

            # Visualize ROIS with GT boxes first
            # img = features[0,0,:,:]
            # plot_boxes(img, gt_boxes[0].tolist(), [], all_rois[0].tolist())
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            
            # Set batch labels to Face based on number of face rois
            labels_batch[i][:fg_rois_per_this_image] = 1

            rois_batch[i] = all_rois[i][keep_inds]

            # TODO: Check why
            # rois_batch[i,:,0] = i

            gt_rois_batch[i] = all_rois[i][gt_assignment[i][keep_inds]]

        bbox_target_data = _compute_targets_pytorch(
                rois_batch, gt_rois_batch)

        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights

def _compute_targets_pytorch(ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        targets = bbox_transform_batch(ex_rois, gt_rois)
        
        # TODO: Explore target normalization
        # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #     # Optionally normalize targets by a precomputed mean and stdev
        #     targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
        #                 / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

def bbox_overlaps_vectorized(anchors, gt_boxes):
    batch_size = gt_boxes.shape[0]

    overlaps = []

    for i in range(batch_size):
        IOUs = anchor_ops.calc_IOU_vectorized(anchors[i], gt_boxes[i])
        IOUs = IOUs.unsqueeze(0)
        overlaps.append(IOUs)
        
    overlaps = torch.cat(overlaps, 0)
    return overlaps 

def bbox_transform_batch(ex_rois, gt_rois):

    targets_dx = (gt_rois[:, :, 0] - ex_rois[:, :, 0])
    targets_dy = (gt_rois[:, :, 1] - ex_rois[:, :, 1])
    targets_dw = (gt_rois[:, :, 2] - ex_rois[:, :, 2])
    targets_dh = (gt_rois[:, :, 3] - ex_rois[:, :, 3])

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets

def _get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights
