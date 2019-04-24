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

        # Step 1.a - Transform anchors shape based on batch size
        boxes_shape = boxes.shape
        boxes = boxes.view(batch_size, boxes_shape[0], boxes_shape[1], boxes_shape[2], boxes_shape[3])

        # Step 1.b - Transform bbox_deltas shape to match the anchor 
        bbox_deltas_shape = bbox_deltas.shape
        split_deltas = bbox_deltas.view(bbox_deltas_shape[0], 16, 4, bbox_deltas_shape[2], bbox_deltas_shape[3])
        split_deltas = split_deltas.view(bbox_deltas_shape[0], 16, bbox_deltas_shape[2], bbox_deltas_shape[3], 4)

        # Step 2 - Apply bounding box transformations
        adjusted_boxes = np.add(boxes.numpy(), split_deltas.numpy())

        # Step 3 - Clip boxes so that they are within the feature dimensions
        clipped_boxes = clip_boxes(adjusted_boxes, height, width, batch_size)

        # Step 4 - Filter those boxes that have dimensions lesser than minimum
        keep = filter_boxes(clipped_boxes, rpn_config.MIN_SIZE)

        # Step 4.a - Flatten and get only boxes and scores that passed the filter
        keep = np.reshape(keep, (batch_size, -1))
        clipped_boxes = np.reshape(clipped_boxes, (batch_size, -1, 4))
        scores = np.reshape(scores.numpy(), (batch_size, -1))

        filtered_boxes = clipped_boxes[keep]
        filtered_scores = scores[keep]

        # TODO: Check if this needs to be changed in case of a batch
        filtered_boxes = np.reshape(filtered_boxes, (batch_size, filtered_boxes.shape[0], filtered_boxes.shape[1]))
        filtered_scores = np.reshape(filtered_scores, (batch_size, filtered_scores.shape[0]))

        # Steps 5 - Sort scores
        filtered_scores = torch.from_numpy(filtered_scores)
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
            scores = scores[order].numpy()

            # Step 7 - Combine anchors and scores
            scores = np.reshape(scores, (scores.shape[0], 1))
            combined = np.concatenate((proposals, scores),axis=1)

            # Step 8 - Apply NMS with a specific threshold in config
            keep_anchors_postNMS = nms(combined, rpn_config.NMS_THRESH)
            print(keep_anchors_postNMS)

            # Step 9 - Take TopN post NMS proposals
            if rpn_config.POST_NMS_TOP_N > 0:
                keep_anchors_postNMS = keep_anchors_postNMS[:rpn_config.POST_NMS_TOP_N]

            proposals = proposals[keep_anchors_postNMS, :]
            scores = scores[keep_anchors_postNMS, :]

            # Step 10 - Return topN proposals as output
            num_proposal = proposals.shape[0]
            output[i,:,0] = i
            output[i, :num_proposal, 0:] = torch.from_numpy(proposals)

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def clip_boxes(boxes, length, width, batch_size):

    for i in range(batch_size):
        for channel in boxes[i]:
            for x in channel:
                for y in x:

                    # Check if the entry exceeds the bounds, else clip
                    y[2] = y[0] + y[2]
                    y[3] = y[1] + y[3]

                    y[0] = np.clip(y[0], 0, length-1)
                    y[1] = np.clip(y[1], 0, length-1)
                    y[2] = np.clip(y[2], 0, length-1)
                    y[3] = np.clip(y[3], 0, length-1)


                    y[2] = y[2] - y[0]
                    y[3] = y[3] - y[1]

    return boxes

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""

    widths = boxes[:, :, :, :, 2]
    heights = boxes[:, :, :, :, 3]
    
    keep = np.zeros_like(widths)
    min_sizes = keep.copy()
    min_sizes.fill(min_size)

    keep = ((widths >= min_sizes) & (heights >= min_sizes))
    return keep

def nms(entries, thresh):

    x1 = entries[:, 0]
    y1 = entries[:, 1]
    l = entries[:, 2]
    b = entries[:, 3]
    scores = entries[:, 4]

    x2 = x1 + l
    y2 = y1 + b    

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
