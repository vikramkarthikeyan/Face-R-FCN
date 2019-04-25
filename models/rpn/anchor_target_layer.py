import torch
from torch import nn
from ..utils.anchors import generate_anchors, calc_IOU
from ..config import rfcn_config as cfg
import numpy as np


class _AnchorLayer(nn.Module):

    def __init__(self):
        super(_AnchorLayer, self).__init__()

    def forward(self, cls_scores, gt_boxes, image_info=None):

        # Algorithm to follow:
        #
        # 1. Generate anchors. All anchors are generated including the ones which exceed boundaries on any
        #       and all sides. Generation function is generic.
        # 2. Clip anchors. All anchors which lie completely within the image boundaries are kept. THis is
        #       done as the anchor generation function is generic for compatibility.
        # 3. Get all area overlap for the kept anchors, including the ones with zero overlap. For mathematical
        #       simplicity, introduce a small offset (1e-5) for the overlap areas which are zero to avoid
        #       possible logical error down the line (ZeroDivisionError).
        # 4. Create labels for all anchors generated and set them as -1.
        #       Label convention adopted:
        #           -1: Don't care.
        #            0: Neg anchor.
        #            1: Pos anchor.
        # 5. Calculate best possible over lap w. r. t. all GT boxes. Choose the best GT box for this match
        # 6. Anchors with overlap area above POS_TH are labeled 1.
        # 7. Anchors with overlap area below NEG_TH are labeled 0.
        #
        # TODO: If number of anchors is too much, subsample from these to get an acceptable number. Change label only.
        #
        # 8. generate the regression targets for the GT boxes for the best overlap from step 5.
        #
        # TODO: inside and outside weights.
        #
        # .

        _, _, height, width = cls_scores.shape

        batch_size = gt_boxes.shape[0]
        # batch_size = 1
        print("\n\n----Proposal Layer----\n")
        print("CLS_SCORES:", cls_scores.shape)
        print("GT_BOXES:", gt_boxes.shape)

        # 1. Generating anchors
        all_anchors = generate_anchors((height, width), cfg.ANCHOR_SIZES)

        all_anchors = all_anchors.view(batch_size, all_anchors.shape[0], all_anchors.shape[1],
                                       all_anchors.shape[2], all_anchors.shape[3])

        print("ANCHORS:", all_anchors.shape)

        # 2. Clipping anchors which are not necessary
        clipped_boxes = self.clip_boxes(all_anchors, height, width, batch_size)
        # clipped_boxes = self.clip_boxes(all_anchors, height, width, batch_size).view(-1, 4)
        # clipped_boxes = clipped_boxes.view(-1)

        print("CLIPPED:", clipped_boxes.shape)

        # 3. Get all area overlap for the kept anchors
        # overlaps = self.bbox_overlaps_batch(torch.tensor(clipped_boxes), torch.tensor(gt_boxes))
        overlaps = self.bbox_overlaps(clipped_boxes, gt_boxes)
        overlaps[overlaps == 0] = 1e-5

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, arggt_max_overlaps = torch.max(overlaps, 1)

        # print("OVERLAPS:", overlaps)
        print("OVERLAPS SHAPE:", overlaps.shape)
        print("OVERLAPS ARGS SHAPE:", argmax_overlaps.shape)
        print("OVERLAPS ARGS:", argmax_overlaps)

        # 4. Create labels for all anchors generated and set them as -1.
        labels = overlaps.new(batch_size, clipped_boxes.shape[0]).fill_(-1)

        # 5. Calculate best possible over lap w. r. t. all GT boxes. Choose the best GT box for this match
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # gt_max_overlaps, arggt_max_overlaps = torch.max(overlaps, 1)

        labels[max_overlaps > cfg.RPN_POSITIVE_OVERLAP] = 1
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0

        targets = clipped_boxes.view(-1, 4) - gt_boxes.view(-1, 4)[argmax_overlaps, :]

        targets = targets.view(batch_size, -1, 4)

        return labels, targets

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    # def bbox_transform_batch(self, ex_rois, gt_rois):
    # # from bbox_transform.py

    def clip_boxes(self, boxes, length, width, batch_size):

        for i in range(batch_size):
            for channel in boxes[i]:
                for x in channel:
                    for y in x:
                        # Check if the entry exceeds the bounds, else clip
                        y[2] = y[0] + y[2]
                        y[3] = y[1] + y[3]

                        y[0] = np.clip(y[0], 0, length - 1)
                        y[1] = np.clip(y[1], 0, length - 1)
                        y[2] = np.clip(y[2], 0, length - 1)
                        y[3] = np.clip(y[3], 0, length - 1)

                        y[2] = y[2] - y[0]
                        y[3] = y[3] - y[1]

        return boxes

    def bbox_overlaps(self, anchors, gt_boxes):
        batch_size = gt_boxes.shape[0]
        overlaps = []

        for i in range(batch_size):
            overlaps_image = []
            for anchor in anchors[i]:
                IOU_anchor_vs_all_gt = [calc_IOU(anchor, gt_box) for gt_box in gt_boxes[i]]

                overlaps_image.append(IOU_anchor_vs_all_gt)

            overlaps.append(overlaps_image)

        return torch.tensor(overlaps)

    def bbox_overlaps_batch(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (b, K, 5) ndarray of float

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        batch_size = gt_boxes.shape[0]

        # if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.shape[1]

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()
        # anchors = anchors.view(1, N, 4).expand(batch_size, N, 4)
        # gt_boxes = gt_boxes[:, :, :4]

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

        return overlaps
