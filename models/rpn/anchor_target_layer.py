import torch
import os
from torch import nn
from ..utils.anchors import generate_anchors, calc_IOU, calc_IOU2, calc_IOU_vectorized_
from ..config import rfcn_config as cfg
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class _AnchorLayer(nn.Module):

    def __init__(self):

        self.all_anchors = None
        self.indices = None

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

        if cfg.verbose:
            print("\n\n----Anchor Target Layer----\n")
            print("GT_BOXES:", gt_boxes.shape)

        # 1. Generating anchors
        if self.all_anchors is None:
            self.all_anchors = generate_anchors((height, width), cfg.ANCHOR_SIZES)

            self.all_anchors = np.reshape(self.all_anchors,
                                          (batch_size, self.all_anchors.shape[0], self.all_anchors.shape[1],
                                           self.all_anchors.shape[2], self.all_anchors.shape[3]))

            if cfg.verbose:
                print("ANCHORS:", self.all_anchors.shape)

            # 2. Clipping anchors which exceed boundaries
            clipped_boxes, indices = self.clip_boxes_batch(self.all_anchors, height, width, batch_size)

            self.clipped_boxes = clipped_boxes
            self.indices = indices

        # 3. Get all area overlap for the kept anchors
        overlaps = self.bbox_overlaps_vectorized(self.clipped_boxes, gt_boxes)

        # Set a minimum value to avoid a potential zero error
        overlaps[overlaps == 0] = 1e-10

        if cfg.verbose:
            print("MAX OF OVERLAPS", overlaps.max())
            print(overlaps[overlaps > 1], "empty is good")
            print(overlaps[overlaps > cfg.RPN_POSITIVE_OVERLAP], "empty is not good")

        # 4. Create labels for all anchors generated and set them as -1.
        # (1, 69151 - 69151 anchors are left after removing those that exceed the boundaries
        labels = np.full((batch_size, self.clipped_boxes.shape[0]), -1)

        # 5. Calculate best possible over lap w. r. t. all GT boxes. Choose the best GT box for this match
        argmax_overlaps = np.argmax(overlaps, 2)
        max_overlaps = np.max(overlaps, 2)

        # 5.a Get the anchor with the best overlap per GT box
        gt_argmax_overlaps = np.argmax(overlaps, 1)

        if cfg.verbose:
            print("OVERLAPS SHAPE:", overlaps.shape)

        # Anchor targets based on threshold
        labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1
        labels[max_overlaps <= cfg.RPN_NEGATIVE_OVERLAP] = 0

        # Positive anchors based on best overlap per GT
        labels[0, gt_argmax_overlaps[0]] = 1

        pos_anc_cnt = np.count_nonzero(labels == 1)
        neg_anc_cnt = np.sum(max_overlaps <= cfg.RPN_NEGATIVE_OVERLAP)

        if cfg.verbose:
            print("positive anchors generated:", pos_anc_cnt)
            print("negative anchors generated:", neg_anc_cnt)

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
            if pos_anc_cnt < 5:
                plot_layer_outputs(self.clipped_boxes, labels, scale, gt_boxes, image_info)

        labels_op = np.full((batch_size, cfg.NUM_ANCHORS * height * width), -1)
        target_op = np.full((batch_size, cfg.NUM_ANCHORS, height, width, 4), 0, dtype=np.float)

        fg_indices_mask = (labels == 1)[0]
        clipped_indices = np.transpose(np.nonzero(self.indices))

        fg_indices = clipped_indices[fg_indices_mask, :]
        gt_assignments = argmax_overlaps[:, fg_indices_mask]

        fg_indices = fg_indices.T
        fg_anchors = self.all_anchors[fg_indices[0], fg_indices[1], fg_indices[2], fg_indices[3], :]
        fg_gt_boxes = gt_boxes[:, gt_assignments[0], :]
        fg_anchors = np.expand_dims(fg_anchors, axis=0)

        bbox_targets = get_bbox_targets(fg_anchors, fg_gt_boxes)

        if cfg.verbose:
            print("FG ANCHORS:", fg_anchors.shape)
            print("FG_GT_BOXES:", fg_gt_boxes.shape)
            print("TARGET_OP:", target_op.shape)
            print("LABELS OP:", labels_op.shape)
            print("BOX_TARGETS:", bbox_targets.shape)
            print("FG INDICES:", fg_indices.shape)

        instance_indices = np.reshape(self.indices, (batch_size, cfg.NUM_ANCHORS * height * width))

        # Transfer labels based on feature map size
        labels_op[0][instance_indices[0]] = labels[0]
        labels_op = np.reshape(labels_op, (batch_size, cfg.NUM_ANCHORS, height, width))


        for i, index_list in enumerate(fg_indices.T):
            target_op[index_list[0], index_list[1], index_list[2], index_list[3], 0] = bbox_targets[0, i, 0]
            target_op[index_list[0], index_list[1], index_list[2], index_list[3], 1] = bbox_targets[0, i, 1]
            target_op[index_list[0], index_list[1], index_list[2], index_list[3], 2] = bbox_targets[0, i, 2]
            target_op[index_list[0], index_list[1], index_list[2], index_list[3], 3] = bbox_targets[0, i, 3]
        
        if cfg.gc_collect:
            del overlaps, argmax_overlaps, max_overlaps, gt_argmax_overlaps

        
        return labels_op, target_op

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

        keep = np.logical_and(keep, boxes <= length - 1)

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


def get_bbox_targets(anchors, gt_boxes):

    targets = np.full(gt_boxes.shape, 0.0, dtype=np.float)

    targets[:, :, 0] = (gt_boxes[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2] 
    targets[:, :, 1] = (gt_boxes[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]

    targets[:, :, 2] = np.log(gt_boxes[:, :, 2] / anchors[:, :, 2])
    targets[:, :, 3] = np.log(gt_boxes[:, :, 3] / anchors[:, :, 3])

    return targets


def resize_image(im, dimension=cfg.IMAGE_INPUT_DIMS):
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


def plot_layer_outputs(clipped_boxes, labels, scale, gt_boxes, image_info):
    pos_anc = clipped_boxes[(labels[0] == 1)]
    neg_anc = clipped_boxes[(labels[0] == 0)]

    img = Image.open(image_info[0])
    # plt.imshow(resize_image(img))

    ax = plt.gca()
    ax.imshow(resize_image(img))

    #for anc in neg_anc[:100, :]:
    #    anc_ = anc * scale
    #    ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
    #                           linewidth=2, edgecolor='r',
    #                           facecolor='none'))

    for anc in pos_anc:
        anc_ = anc * scale
        ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
                               linewidth=2, edgecolor='g',
                               facecolor='none'))

    for anc in gt_boxes[0]:
        anc_ = anc * scale
        ax.add_patch(Rectangle((anc_[0], anc_[1]), anc_[2], anc_[3],
                               linewidth=2, edgecolor='b',
                               facecolor='none'))

    file_name = image_info[0].split('/')[-1]

    cwd = os.getcwd()

    plt.savefig(cwd + "/debug_plots/anchor_target_layer/" + file_name)
