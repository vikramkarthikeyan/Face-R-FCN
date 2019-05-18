import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import time

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
        # self.anchors = anchors.generate_anchors((height, width), self.box_sizes)
        self.anchors = None
        self.scale = scale
        self.boxes = None

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
        
        if self.anchors is None:
            self.anchors = anchors.generate_anchors((height, width), self.box_sizes)
            self.boxes = self.anchors
            
            # Step 1.a - Transform anchors shape to match the batch size
            boxes_shape = self.boxes.shape
            self.boxes = np.reshape(self.boxes, (batch_size, boxes_shape[0], boxes_shape[1], boxes_shape[2], boxes_shape[3]))

            self.boxes = torch.from_numpy(self.boxes)
            
        
        # Step 1.b - Transform bbox_deltas shape to match the anchor shape
        bbox_deltas_shape = bbox_deltas.shape

        split_deltas = bbox_deltas.view(bbox_deltas_shape[0], rfcn_config.NUM_ANCHORS, 4, bbox_deltas_shape[2],
                                        bbox_deltas_shape[3])
                                        
        split_deltas = split_deltas.view(bbox_deltas_shape[0], rfcn_config.NUM_ANCHORS, bbox_deltas_shape[2],
                                         bbox_deltas_shape[3], 4)

        # boxes shape: 1,20,64,64,4
        # split_deltas shape: 1,20,64,64,4
        # face scores shape: 1,20,64,64 

        # Step 2 - Apply bounding box transformations
        adjusted_boxes = bbox_transform(self.boxes, split_deltas)
        
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

        filtered_boxes = filtered_boxes.view(batch_size, filtered_boxes.shape[0], filtered_boxes.shape[1])
        filtered_scores = filtered_scores.view(batch_size, filtered_scores.shape[0])
        
        # Steps 5 - Sort scores
        _, orders = torch.sort(filtered_scores, 1, True)

        # Create output array for RPN results
        proposal_outputs = torch.zeros((batch_size, rpn_config.POST_NMS_TOP_N, 4), dtype=torch.float)
        score_outputs = torch.zeros((batch_size, rpn_config.POST_NMS_TOP_N, 1), dtype=torch.float)
        
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

            # Step 8 - Apply NMS with a specific threshold in config
            keep_anchors_postNMS = nms_numpy(proposals, rpn_config.NMS_THRESH)

            # Step 9 - Take TopN post NMS proposals
            if rpn_config.POST_NMS_TOP_N > 0:
                keep_anchors_postNMS = keep_anchors_postNMS[:rpn_config.POST_NMS_TOP_N]

            proposals = proposals[keep_anchors_postNMS, :]
            scores = scores[keep_anchors_postNMS]

            if rfcn_config.verbose:
                print("\nPOST NMS SIZE:", proposals.shape, scores.shape)

            # Step 10 - Return topN proposals as proposal_outputs
            num_proposal = proposals.shape[0]
            proposal_outputs[i, :num_proposal, 0:] = proposals
            score_outputs[i, :num_proposal, 0] = scores
        
        return proposal_outputs, score_outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def bbox_transform(boxes, split_deltas):
    results =  torch.zeros_like(boxes).float()
    
    results[:,:,:,:,0] = (split_deltas[:,:,:,:,0] * boxes[:,:,:,:,0]) + boxes[:,:,:,:,0]
    results[:,:,:,:,1] = (split_deltas[:,:,:,:,1] * boxes[:,:,:,:,1]) + boxes[:,:,:,:,1]
    results[:,:,:,:,2] = torch.exp(split_deltas[:,:,:,:,2]) * boxes[:,:,:,:,2]
    results[:,:,:,:,3] = torch.exp(split_deltas[:,:,:,:,3]) * boxes[:,:,:,:,3]
    return results
    

def clip_boxes_batch(boxes, length, width, batch_size):
    """
    Clip boxes to image boundaries.
    """

    #boxes = np.nan_to_num(boxes)
    boxes[boxes < 0] = 0

    boxes[:, :, :, :, 2] = boxes[:, :, :, :, 0] + boxes[:, :, :, :, 2] - 1
    boxes[:, :, :, :, 3] = boxes[:, :, :, :, 1] + boxes[:, :, :, :, 3] - 1

    boxes[:, :, :, :, 0][boxes[:, :, :, :, 0] > length - 1] = length - 1
    boxes[:, :, :, :, 1][boxes[:, :, :, :, 1] > width - 1] = width - 1
    boxes[:, :, :, :, 2][boxes[:, :, :, :, 2] > length - 1] = length - 1
    boxes[:, :, :, :, 3][boxes[:, :, :, :, 3] > width - 1] = width - 1

    boxes[:, :, :, :, 2] = boxes[:, :, :, :, 2] - boxes[:, :, :, :, 0] + 1
    boxes[:, :, :, :, 3] = boxes[:, :, :, :, 3] - boxes[:, :, :, :, 1] + 1

    return boxes

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""

    widths = boxes[:, :, :, :, 2]
    heights = boxes[:, :, :, :, 3]

    keep = torch.zeros_like(widths)

    min_sizes = keep.new_full(keep.shape, min_size)

    keep = ((widths >= min_sizes) & (heights >= min_sizes))
    return keep

def nms_numpy(entries, thresh):
    x1 = entries[:, 0]
    y1 = entries[:, 1]
    l = entries[:, 2]
    b = entries[:, 3]
    # COMMENTED FOR SPEED
    # TODO: check why it is needed
    # scores = entries[:, 4]

    x2 = x1 + l - 1
    y2 = y1 + b - 1

    # Initialize list of picked indices
    idx_keep = []

    # Calculate areas of all bounding boxes
    areas = l * b

    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    _, idxs = torch.sort(y2)
    # idxs = np.argsort(y2)

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

        mask = ((W * H).float() / areas.float()).lt(thresh)
        print(mask)
        # mask = ((W * H) / areas) < thresh
        mask[-1] = False

        areas = areas[mask]
        idxs = idxs[mask]

    # return only the bounding boxes that were picked
    return torch.tensor(idx_keep)
