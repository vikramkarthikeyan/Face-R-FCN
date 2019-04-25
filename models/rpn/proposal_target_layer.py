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


BBOX_INSIDE_WEIGHTS = torch.FloatTensor(rfcn_config.BBOX_INSIDE_WEIGHTS)

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, n_classes):
        super(_ProposalTargetLayer, self).__init__()
        self.num_classes = n_classes
        
    def forward(self, rois, gt_boxes):

        batch_size = rois.shape[0]

        gt_boxes = torch.from_numpy(gt_boxes).float()

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([rois, gt_boxes], 1)

        print("\n----Proposal Target Layer----\n\nTotal ROIs with GT boxes:", all_rois.shape)

        rois_per_image = rfcn_config.ROI_BATCH_SIZE

        fg_rois_per_image = int(np.round(rfcn_config.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # Generate examples using OHEM
        data = _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, self.num_classes)
        


        return all_rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # calculate all combinations of overlaps (rois x gt_boxes)
        overlaps = bbox_overlaps(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 1)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

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

            print("\nFace and BG ROIs:", num_fg_rois, bg_num_rois)

            if num_fg_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, num_fg_rois)
                rand_num = torch.from_numpy(np.random.permutation(num_fg_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif num_fg_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(rois_per_image) * num_fg_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]

                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and num_fg_rois == 0:
                # sampling bg
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and num_fg_rois = 0, this should not happen!")
                
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

def bbox_overlaps(anchors, gt_boxes):
    batch_size = anchors.shape[0]
    overlaps = []

    for i in range(batch_size):
        overlaps_image = []
        for anchor in anchors[i]:
            IOU_anchor_vs_all_gt = [anchor_ops.calc_IOU(anchor, gt_box) for gt_box in gt_boxes[i]]

            overlaps_image.append(IOU_anchor_vs_all_gt)
    
        overlaps.append(overlaps_image)

    return torch.tensor(overlaps)

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

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