import torch
from torch import nn
from ..utils.anchors import generate_anchors, calc_IOU, calc_IOU2, calc_IOU_vectorized_
from ..config import rfcn_config as cfg
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class _AnchorLayer(nn.Module):

    def __init__(self):

        self.anchors = None

        super(_AnchorLayer, self).__init__()

    def forward(self, cls_scores, gt_boxes, image_info):

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
        # DONE: If number of anchors is too much, subsample from these to get an acceptable number. Change label only.
        #
        # 8. generate the regression targets for the GT boxes for the best overlap from step 5.

        _, _, height, width = cls_scores.shape
        scale = cfg.IMAGE_INPUT_DIMS // height

        batch_size = gt_boxes.shape[0]

        cls_scores = cls_scores.cpu()

        if cfg.verbose:
            print("\n\n----Anchor Target Layer----\n")
            print("CLS_SCORES:", cls_scores.shape)
            print("GT_BOXES:", gt_boxes.shape, gt_boxes)

        # 1. Generating anchors
        if self.anchors is None:
            self.anchors = generate_anchors((height, width), cfg.ANCHOR_SIZES)

            all_anchors = self.anchors

            all_anchors = np.reshape(all_anchors, (batch_size, all_anchors.shape[0], all_anchors.shape[1],
                                       all_anchors.shape[2], all_anchors.shape[3]))

            if cfg.verbose:
                print("ANCHORS:", all_anchors.shape)

            # 2. Clipping anchors which exceed boundaries
            clipped_boxes, indices = self.clip_boxes_batch(all_anchors, height, width, batch_size)
            
            self.clipped_boxes = clipped_boxes
            self.indices = indices

            if cfg.verbose:
                print("CLIPPED:", clipped_boxes.shape, clipped_boxes)


        # 3. Get all area overlap for the kept anchors
        overlaps = self.bbox_overlaps_vectorized(self.clipped_boxes, gt_boxes)

        # Set a minimum value to avoid a potential zero error
        overlaps[overlaps == 0] = 1e-10
        
        if cfg.verbose:
            print("MAX OF OVERLAPS", overlaps.max())
            print(overlaps[overlaps > 1], "empty is good")
            print(overlaps[overlaps > cfg.RPN_POSITIVE_OVERLAP], "empty is not good")

        # 4. Create labels for all anchors generated and set them as -1.
        labels = np.full((batch_size, self.clipped_boxes.shape[0]), -1)

        # 5. Calculate best possible over lap w. r. t. all GT boxes. Choose the best GT box for this match
        argmax_overlaps = np.argmax(overlaps, 2)
        max_overlaps = np.max(overlaps, 2)
        
        #max_overlaps, argmax_overlaps = torch.max(overlaps, 2)

        if cfg.verbose:
            print("OVERLAPS SHAPE:", overlaps.shape)

        labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1
        labels[max_overlaps <= cfg.RPN_NEGATIVE_OVERLAP] = 0

        pos_anc_cnt = np.sum(max_overlaps >= cfg.RPN_POSITIVE_OVERLAP)
        neg_anc_cnt = np.sum(max_overlaps <= cfg.RPN_NEGATIVE_OVERLAP)
        
        if cfg.verbose:
            print("positive anchors generated:", pos_anc_cnt)
        
        total_rois = 256
        num_acceptable_fg = total_rois // 3
        num_acceptable_bg = (total_rois * 2) // 3

        # Subsample if there are too many bg anchors
        if neg_anc_cnt > num_acceptable_bg:
            for i in range(batch_size):
                bg_inds = np.transpose(np.nonzero(labels[i] == 0))
                rand_num = np.random.permutation(bg_inds.shape[0])
                disable_inds = bg_inds[rand_num[num_acceptable_bg:]]
                labels[i][disable_inds] = -1
        
        # Subsample if there are too many fg anchors
        if pos_anc_cnt > num_acceptable_fg:
            for i in range(batch_size):
                fg_inds = np.transpose(np.nonzero(labels[i] == 1))
                rand_num = np.random.permutation(fg_inds.shape[0])
                disable_inds = fg_inds[rand_num[num_acceptable_fg:]]
                labels[i][disable_inds] = -1

        if cfg.demo:
            plot_layer_outputs(clipped_boxes, labels, scale, image_info)

        targets = np.zeros((batch_size, self.clipped_boxes.shape[0], 4), dtype=np.float32)
        #TODO: Convert target generation to log: http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/ 
        label_op = overlaps.new(batch_size, cfg.NUM_ANCHORS, height, width, 1).fill_(-1)
        target_op = overlaps.new(batch_size, cfg.NUM_ANCHORS, height, width, 4).fill_(0)

        if cfg.verbose:
            print("TARGET_IP:", targets.shape)
            print("TARGET_OP:", target_op.shape)

        for lab, target, ind in zip(labels[0], targets[0], self.indices):
            label_op[ind] = lab
            target_op[ind] = target

        return label_op, target_op

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def plot_boxes(self, anchors, color='r'):
        pass

    def clip_boxes_batch(self, boxes, length, width, batch_size):
        """
        Clip boxes to image boundaries.
        """
        keep = boxes >= 0

        boxes[:, :, :, :, 2] = boxes[:, :, :, :, 0] + boxes[:, :, :, :, 2] - 1
        boxes[:, :, :, :, 3] = boxes[:, :, :, :, 1] + boxes[:, :, :, :, 3] - 1
        
        keep = np.logical_and(keep, boxes<=length-1)
        
        keep = np.all(keep, axis=4) 
        
        boxes[:, :, :, :, 2] = boxes[:, :, :, :, 2] - boxes[:, :, :, :, 0] + 1
        boxes[:, :, :, :, 3] = boxes[:, :, :, :, 3] - boxes[:, :, :, :, 1] + 1
        
        boxes = boxes[keep, :]

        return boxes, keep

    def bbox_overlaps_vectorized(self, anchors, gt_boxes):
        batch_size = gt_boxes.shape[0]

        overlaps = []

        for i in range(batch_size):
            IOUs = calc_IOU_vectorized_(anchors, gt_boxes[i])
            overlaps.append(IOUs)

        return np.array(overlaps)


def resize_image(self, im, dimension=cfg.IMAGE_INPUT_DIMS):
    old_size = im.size
    ratio = float(dimension) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    offset_x = (dimension - new_size[0]) // 2
    offset_y = (dimension - new_size[1]) // 2

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (dimension, dimension))
    new_im.paste(im, (offset_x, offset_y))

    return new_im


def plot_layer_outputs(clipped_boxes, labels, scale, image_info):
    pos_anc = clipped_boxes[(labels == 1).view(-1), :]
    neg_anc = clipped_boxes[(labels == 0).view(-1), :]

    img = Image.open(image_info[0])
    plt.imshow(resize_image(img))

    ax = plt.gca()

    for anc in neg_anc[:100, :]:
        anc_ = anc * scale
        ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
                               linewidth=2, edgecolor='r',
                               facecolor='none'))

    for anc in pos_anc:
        anc_ = anc * scale
        ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
                               linewidth=2, edgecolor='g',
                               facecolor='none'))

    for anc in gt_boxes[0]:
        anc_ = anc
        ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
                               linewidth=2, edgecolor='b',
                               facecolor='none'))

    plt.show()
