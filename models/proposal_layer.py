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
        positive_anchors, negative_anchors = get_regions(features, 100, b_boxes, self._box_sizes)

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
    


def get_regions(features, N, list_bb, box_size):
    """
    Function to generate Positive and Negative anchors.
    :param features:  Extracted features
    :param N:         Number of bounding boxes
    :param list_bb:   Bounding boxes (scaled to current configuration)
    :return:          Positive and negative anchors.
    """

    # box_size = [(32, 32), (64, 32), (32, 64)]
    # box_size = [(128, 128), (256, 128), (128, 256), (256, 256), (256, 512), (512, 256)]
    """Sizes for region proposals"""
    _, _, feat_h, feat_w = features.shape

    print("Shape of features:", features.shape)
    print("Shape of bounding box:", list_bb[0])

    pos_anc = []
    neg_anc = []
    scale = 1

    for bs in box_size:
        for i in range(0, feat_h):
            for j in range(0, feat_w):

                max_iou = 0

                x = min(feat_h, max(0, (i - (bs[0] // 2))))
                y = min(feat_w, max(0, (j - (bs[1] // 2))))

                # im_slice = features[x:x + bs[0], y:y + bs[1]]
                frame_a = (x, y, x + bs[0], y + bs[1])
                """
                Frame A is the sliding window for calculation.
                """

                for bb in list_bb:
                    frame_b = (bb[1], bb[0], bb[1] + bb[3], bb[0] + bb[2])
                    """
                    Frame B is the input """
                    iou = calc_IOU(frame_a, frame_b)
                    if max_iou < iou:
                        max_iou = iou
                
                    # print(max_iou)

                if max_iou > 0.7:
                    print(frame_a, frame_b, max_iou)
                    # pos_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))
                    pos_anc.append([y * scale, x * scale,  bs[1], bs[0]])
                elif max_iou < 0.05:
                    # neg_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))
                    neg_anc.append([y * scale, x * scale,  bs[1], bs[0]])

    return pos_anc, neg_anc[:100]


def calc_IOU(boxA, boxB):
    """
    Function taken from this site:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param boxA: (X_min, Y_min, X_max, Y_max)
    :param boxB: (X_min, Y_min, X_max, Y_max)
    :return: IOU area ratio.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def plot_boxes(im, positive_anchors, negative_anchors, boxes):
    # torchvision.transforms.ToPILImage().

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
