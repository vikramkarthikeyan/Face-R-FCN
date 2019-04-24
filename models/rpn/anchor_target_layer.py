import torch.nn as nn


class _AnchorLayer(nn.Module):

    def __init__(self):
        super(_AnchorLayer, self).__init__()

    def forward(self, cls_scores, gt_boxes, image_info=None):

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
        # 5. Anchors with overlap area above POS_TH are labeled 1.
        # 6. Anchors with overlap area below NEG_TH are labeled 0.
        #
        # TODO: If number of anchors is too much, subsample from these to get an acceptable number. Change label only.
        #
        # 7, Associate
        #
        #
        #
        # 
        # .

        _, _, height, width = cls_scores.shape

        print("CLS_SCORES:", cls_scores.shape)
        print("GT_BOXES:", gt_boxes.shape)

        return 0, 0

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
