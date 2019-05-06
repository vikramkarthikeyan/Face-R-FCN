import torch
from torch import nn
from ..utils.anchors import generate_anchors, calc_IOU, calc_IOU2
from ..config import rfcn_config as cfg
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


class _AnchorLayer(nn.Module):

    def __init__(self):
        super(_AnchorLayer, self).__init__()

    def forward(self, cls_scores, gt_boxes_old, image_info=None):

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
        #
        # TODO: inside and outside weights.
        #
        # .

        _, _, height, width = cls_scores.shape
        scale = cfg.IMAGE_INPUT_DIMS // height

        flag_verbose = True
        # flag_verbose = False
        flag_demo = True
        # flag_demo = False

        batch_size = gt_boxes_old.shape[0]
        # batch_size = 1

        gt_boxes = torch.tensor(gt_boxes_old)

        if flag_verbose:
            print("\n\n----Anchor Target Layer----\n")
            print("CLS_SCORES:", cls_scores.shape)
            print("GT_BOXES:", gt_boxes.shape, gt_boxes)

        # 1. Generating anchors
        all_anchors = generate_anchors((height, width), cfg.ANCHOR_SIZES)

        all_anchors = all_anchors.view(batch_size, all_anchors.shape[0], all_anchors.shape[1],
                                       all_anchors.shape[2], all_anchors.shape[3])

        if flag_verbose:
            print("ANCHORS:", all_anchors.shape)

        # 2. Clipping anchors which are not necessary
        clipped_boxes, indices = self.clip_boxes(all_anchors, height, width, batch_size)

        if flag_verbose:
            print("CLIPPED:", clipped_boxes.shape, clipped_boxes)

        # 3. Get all area overlap for the kept anchors
        overlaps = self.bbox_overlaps(clipped_boxes, gt_boxes)
        overlaps[overlaps == 0] = 1e-10

        if flag_verbose:
            print("MAX OF OVERLAPS", overlaps.max())
            print(overlaps[overlaps > 1], "empty is good")
            print(overlaps[overlaps > cfg.RPN_POSITIVE_OVERLAP], "empty is not good")

        # 4. Create labels for all anchors generated and set them as -1.
        labels = overlaps.new(batch_size, clipped_boxes.shape[0]).fill_(-1)
        print("LABELS NEWLY CREATED:", labels.shape)

        # 5. Calculate best possible over lap w. r. t. all GT boxes. Choose the best GT box for this match
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, argmax_gt_max_overlaps = torch.max(overlaps, 1)

        print("GT_MAX:", gt_max_overlaps, argmax_gt_max_overlaps)

        if flag_verbose:
            # print("OVERLAPS:", overlaps)
            print("OVERLAPS SHAPE:", overlaps.shape)
            print("OVERLAPS MAX SHAPE:", max_overlaps.shape)
            print("OVERLAPS ARGS SHAPE:", argmax_overlaps.shape)
            print("OVERLAPS ARGS:", argmax_overlaps)

        labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1
        labels[max_overlaps <= cfg.RPN_NEGATIVE_OVERLAP] = 0

        pos_anc_cnt = torch.sum(max_overlaps > cfg.RPN_POSITIVE_OVERLAP)

        cutoff_cnt = max(1, 3 * pos_anc_cnt)

        if torch.sum(max_overlaps < cfg.RPN_NEGATIVE_OVERLAP) > cutoff_cnt:
            for i in range(batch_size):
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.shape[0])).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0) - cutoff_cnt]]
                labels[i][disable_inds] = -1

        if flag_demo:

            pos_anc = clipped_boxes[(labels == 1).view(-1), :]
            neg_anc = clipped_boxes[(labels == 0).view(-1), :]

            img = Image.open(image_info[0])
            plt.imshow(self.resize_image(img))
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

        targets = clipped_boxes.view(-1, 4).float() - (gt_boxes.view(-1, 4)[argmax_overlaps, :].float() / scale)


        label_op = overlaps.new(batch_size, cfg.NUM_ANCHORS, height, width, 1).fill_(-1)
        target_op = overlaps.new(batch_size, cfg.NUM_ANCHORS, height, width, 4).fill_(0)

        if flag_verbose:
            print("LABEL_ip:", labels.shape)
            print("LABEL_OP:", label_op.shape)
            print("TARGET_IP:", targets.shape)
            print("TARGET_OP:", target_op.shape)

        for lab, target, ind in zip(labels[0], targets[0], indices):
            label_op[ind] = lab
            target_op[ind] = target
            # print(".", end="")

        return label_op, target_op

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    # def bbox_transform_batch(self, ex_rois, gt_rois):
    # # from bbox_transform.py
    def plot_boxes(self, anchors, color='r'):
        pass

    def clip_boxes(self, boxes, length, width, batch_size):

        output = []
        op2 = []
        inds = []

        for i in range(batch_size):
            new_batch = []
            for ch, channel in enumerate(boxes[i]):
                new_channel = []
                for x_n, x in enumerate(channel):
                    new_x = []
                    for y_n, y in enumerate(x):

                        if y[0] >= 0 and y[1] >= 0 and (y[0] + y[2]) < length and (y[1] + y[3]) < width:
                            op2.append(y.numpy())
                            inds.append((i, ch, x_n, y_n))

        return torch.from_numpy(np.array(op2)), inds
        # return boxes

    def bbox_overlaps(self, anchors, gt_boxes, factor=16):
        batch_size = gt_boxes.shape[0]
        overlaps = []

        for i in range(batch_size):
            overlaps_image = []
            for anchor in anchors:
                IOU_anchor_vs_all_gt = [calc_IOU(anchor * factor, gt_box) for gt_box in gt_boxes[i]]

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
