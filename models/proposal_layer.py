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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import yaml
import pdb
import torchvision

from PIL import Image
from .anchors import generate_anchors



DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, box_sizes, scale):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._box_sizes = box_sizes
        self._scale = scale

    def forward(self, features, b_boxes, im_path, image):

        orig_b_boxes = b_boxes

        # Convert bounding boxes to a scale of the feature map from the input image
        b_boxes = scale_boxes(b_boxes, self._scale, 'down')
        
        # Generate anchors
        positive_anchors, negative_anchors = generate_anchors(features, 100, b_boxes, self._box_sizes)

        # Convert anchors to original image size
        b_boxes = scale_boxes(b_boxes, self._scale, 'up')
        positive_anchors = scale_boxes(positive_anchors, self._scale, 'up')
        negative_anchors = scale_boxes(negative_anchors, self._scale, 'up')


        # Plot anchors on original image
        plot_boxes(image[0], positive_anchors, negative_anchors, b_boxes)

        return []

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def scale_boxes(boxes, scale, scale_type):
    results = []

    if scale_type == 'down':
        scale = 1/scale

    for box in boxes:
        x = int(float(box[0]) * scale)
        y = int(float(box[1]) * scale)
        l = int(float(box[2]) * scale)
        b = int(float(box[3]) * scale)
        results.append([x,y,l,b])
    
    return results

def plot_boxes(im, positive_anchors, negative_anchors, boxes):

    image = torchvision.transforms.ToPILImage()(im)
    im = np.array(image, dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    if positive_anchors:
        for i, box in enumerate(positive_anchors):
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='green',facecolor='none')
            ax.add_patch(rect)
    
    if negative_anchors:
        for i, box in enumerate(negative_anchors):
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    for i, box in enumerate(boxes):
        rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    plt.show()
