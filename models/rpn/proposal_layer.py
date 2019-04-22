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

    def forward(self, scores, bbox_deltas, image_info):

        batch_size = 1

        # Algorithm
        # 1. At each location i (h,w) generate A anchors of different scales and ratios
        # 2. Apply predicted bounding box deltas to each of the A anchors
        # 3. Clip adjusted bounding boxes so that they are within the feature dimensions
        # 4. Remove those predicted bounding boxes that have a size lesser than some threshold
        # 5. Associate each bounding box with the scores predicted
        # 6. Sort all (bounding box, score) pairs by score from highest to lowest
        # 7. Take top pre_nms_topN proposals before NMS
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

        # Steps 5,6 - Sort scores and anchors
        filtered_scores = torch.from_numpy(filtered_scores)
        _, orders = torch.sort(filtered_scores, 1, True)
        
        # Create output array for RPN results
        output = filtered_scores.new(batch_size, rpn_config.POST_NMS_TOP_N, 5).zero_()

        for i in range(batch_size):
            proposals = filtered_boxes[i]
            scores = filtered_scores[i]
            order = orders[i]
            
            # Step 7 - Take top pre_nms_topN proposals before NMS
            if rpn_config.PRE_NMS_TOP_N > 0:
                order = order[:rpn_config.PRE_NMS_TOP_N]

            proposals = proposals[order, :]
            scores = scores[order].view(-1,1)

            # Step 8 - Apply NMS with a specific threshold in config

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