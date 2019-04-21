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

        # Algorithm
        # 1. At each location i (h,w) generate A anchors of different scales and ratios
        # 2. Apply predicted bounding box deltas to each of the A anchors
        # 3. Crop adjusted bounding boxes so that they are within the feature dimensions
        # 4. Remove those predicted bounding boxes that have a size lesser than some threshold
        # 5. Associate each bounding box with the scores predicted
        # 6. Sort all (bounding box, score) pairs by score from highest to lowest
        # 7. Take top pre_nms_topN proposals before NMS
        # 8. Apply NMS with threshold 0.7 to remaining proposals
        # 9. Take after_nms_topN proposals after NMS
        # 10. Return the top proposals (-> RoIs top, scores top)

        # Step 1
        _, _, height, width = scores.shape
        boxes = anchors.generate_anchors((height, width), self.box_sizes)

        # Step 2
        boxes = transform_boxes(boxes, bbox_deltas)


        return []

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def transform_boxes(boxes, deltas):
    print(boxes.shape)
    print(deltas.shape)

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes