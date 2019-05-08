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
from ..config import rfcn_config


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, box_sizes, scale):
        super(_ProposalLayer, self).__init__()

        self.feat_stride = feat_stride
        self.box_sizes = box_sizes
        self.scale = scale

    def forward(self, scores, bbox_deltas, image_metadata):

        batch_size = 1

        # Algorithm
        # 1. At each location i (h,w) generate A anchors of different scales and ratios
        # 2. Apply predicted bounding box deltas to each of the A anchors
        # 3. Clip adjusted bounding boxes so that they are within the feature dimensions
        # 4. Remove those predicted bounding boxes that have a size lesser than some threshold
        # 5. Order all scores from highest to lowest
        # 6. Take top pre_nms_topN proposals before NMS by applying score orders on proposals
        # 7. Combine anchors and scores in single representation
        # 8. Apply NMS with threshold 0.7 to remaining proposals
        # 9. Take after_nms_topN proposals after NMS
        # 10. Return the top proposals (-> RoIs top, scores top)

        # Step 1 - Generate Anchors
        _, _, height, width = scores.shape
        boxes = anchors.generate_anchors((height, width), self.box_sizes)
        boxes = boxes.cuda()

        # Step 1.a - Transform anchors shape based on batch size
        boxes_shape = boxes.shape
        boxes = boxes.view(batch_size, boxes_shape[0], boxes_shape[1], boxes_shape[2], boxes_shape[3])

        # Step 1.b - Transform bbox_deltas shape to match the anchor 
        bbox_deltas_shape = bbox_deltas.shape
        
        split_deltas = bbox_deltas.view(bbox_deltas_shape[0], rfcn_config.NUM_ANCHORS, 4, bbox_deltas_shape[2],
                                        bbox_deltas_shape[3])
        split_deltas = split_deltas.view(bbox_deltas_shape[0], rfcn_config.NUM_ANCHORS, bbox_deltas_shape[2],
                                         bbox_deltas_shape[3], 4)
        
        # Step 2 - Apply bounding box transformations
        adjusted_boxes = boxes + split_deltas

        # Step 3 - Clip boxes so that they are within the feature dimensions
        clipped_boxes = clip_boxes_batch(adjusted_boxes, height, width, batch_size)

        # Step 4 - Filter those boxes that have dimensions lesser than minimum
        keep = filter_boxes(clipped_boxes, rpn_config.MIN_SIZE)
        
        # Step 4.a - Flatten and get only boxes and scores that passed the filter
        keep = keep.view(batch_size, -1)
        clipped_boxes = clipped_boxes.view(batch_size, -1, 4)
        scores = scores.view(batch_size, -1)
        
        filtered_boxes = clipped_boxes[keep]
        filtered_scores = scores[keep]
        
        # TODO: Check if this needs to be changed in case of a batch
        filtered_boxes = filtered_boxes.view(batch_size, filtered_boxes.shape[0], filtered_boxes.shape[1])
        filtered_scores = filtered_scores.view(batch_size, filtered_scores.shape[0])

        # Steps 5 - Sort scores
        _, orders = torch.sort(filtered_scores, 1, True)

        # Create output array for RPN results
        output = filtered_scores.new(batch_size, rpn_config.POST_NMS_TOP_N, 4).zero_()

        for i in range(batch_size):
            proposals = filtered_boxes[i]
            scores = filtered_scores[i]
            order = orders[i]

            # Step 6 - Take top pre_nms_topN proposals before NMS
            if rpn_config.PRE_NMS_TOP_N > 0:
                order = order[:rpn_config.PRE_NMS_TOP_N]

            # Step 6.a - Filter those topN anchors and scores based on sorted scores
            proposals = proposals[order, :]
            scores = scores[order]

            if rfcn_config.verbose:
                print("\n----Proposal Layer----\n\nPRE NMS SIZE:", proposals.shape)

            # Step 7 - Combine anchors and scores
            scores = scores.view(scores.shape[0], 1)
            combined = torch.cat((proposals, scores), dim=1)
            
            # Step 8 - Apply NMS with a specific threshold in config
            keep_anchors_postNMS = nms(combined, rpn_config.NMS_THRESH)

            #keep_anchors_postNMS = nms_old(combined, rpn_config.NMS_THRESH)
            
            # Step 9 - Take TopN post NMS proposals
            if rpn_config.POST_NMS_TOP_N > 0:
                keep_anchors_postNMS = keep_anchors_postNMS[:rpn_config.POST_NMS_TOP_N]

            proposals = proposals[keep_anchors_postNMS, :]
            scores = scores[keep_anchors_postNMS, :]

            if rfcn_config.verbose:
                print("\nPOST NMS SIZE:", proposals.shape)

            # Step 10 - Return topN proposals as output
            num_proposal = proposals.shape[0]
            output[i, :, 0] = i
            output[i, :num_proposal, 0:] = proposals

        output = output[:, :num_proposal, ]
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def clip_boxes_batch(boxes, length, width, batch_size):
    """
    Clip boxes to image boundaries.
    """

    boxes[boxes < 0] = 0

    # print(boxes[:,:,:,:,0])

    boxes[:,:,:,:,2] = boxes[:,:,:,:,0] + boxes[:,:,:,:,2] - 1
    boxes[:,:,:,:,3] = boxes[:,:,:,:,1] + boxes[:,:,:,:,3] - 1

    boxes[:,:,:,:,0][boxes[:,:,:,:,0] > length - 1] = length - 1
    boxes[:,:,:,:,1][boxes[:,:,:,:,1] > width - 1] = width - 1
    boxes[:,:,:,:,2][boxes[:,:,:,:,2] > length - 1] = length - 1
    boxes[:,:,:,:,3][boxes[:,:,:,:,3] > width - 1] = width - 1

    boxes[:,:,:,:,2] = boxes[:,:,:,:,2] - boxes[:,:,:,:,0] + 1
    boxes[:,:,:,:,3] = boxes[:,:,:,:,3] - boxes[:,:,:,:,1] + 1

    return boxes

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""

    widths = boxes[:, :, :, :, 2]
    heights = boxes[:, :, :, :, 3]

    keep = torch.zeros_like(widths)

    min_sizes = keep.new_full(keep.shape, min_size)
    
    keep = ((widths >= min_sizes) & (heights >= min_sizes))
    return keep


# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
def nms_old(entries, thresh):

    x1 = entries[:, 0]
    y1 = entries[:, 1]
    l = entries[:, 2]
    b = entries[:, 3]
    scores = entries[:, 4]

    x2 = x1 + l - 1
    y2 = y1 + b - 1

    # Initialize list of picked indices
    keep = []

    # Calculate areas of all bounding boxes
    areas = l * b

    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
        last = len(idxs) - 1
        i = idxs[last]
        keep.append(i)
        suppress = [last]
        
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / areas[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > rpn_config.NMS_THRESH:
                suppress.append(pos)
 
        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
 
    # return only the bounding boxes that were picked
    return keep


# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
def nms(entries, thresh):
    x1 = entries[:, 0]
    y1 = entries[:, 1]
    l = entries[:, 2]
    b = entries[:, 3]
    scores = entries[:, 4]

    x2 = x1 + l - 1
    y2 = y1 + b - 1

    # Initialize list of picked indices
    idx_keep = []

    # Calculate areas of all bounding boxes
    areas = l * b

    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    _, idxs = torch.sort(y2)

    # keep looping while some indexes still remain in the indexes list
    while idxs.shape[0] > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        ind = idxs[-1]
        idx_keep.append(ind)

        XX1 = torch.clamp(x1[idxs], min=x1[ind])
        YY1 = torch.clamp(y1[idxs], min=y1[ind])

        XX2 = torch.clamp(x2[idxs], max=x2[ind])
        YY2 = torch.clamp(y2[idxs], max=y2[ind])
   
        W = torch.clamp((XX2 - XX1 + 1), min=0)
        H = torch.clamp((YY2 - YY1 + 1), min=0)

        mask = ((W * H).float() / areas.float()).lt(rpn_config.NMS_THRESH)
        mask[-1] = 0

        areas = areas[mask]
        idxs = idxs[mask]

    # return only the bounding boxes that were picked
    return torch.tensor(idx_keep)
