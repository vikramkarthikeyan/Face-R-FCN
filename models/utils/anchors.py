import numpy as np
import torch


# def generate_anchors(dimensions, box_sizes):
#     """
#     Function to generate anchors.
#     :param dimensions:  Dimensions of extracted features
#     :return:         A*i Anchors (A anchors per location i)
#     """
#     # print("Input dimensions:", dimensions)
#     feat_h, feat_w = dimensions

#     anchors_list = []

#     for bs in box_sizes:
#         x_list = []
#         for i in range(0, feat_h):
#             y_list = []
#             for j in range(0, feat_w):

#                 x = (i - (bs[0] // 2))
#                 y = (j - (bs[1] // 2))
#                 l = x + bs[0]
#                 w = y + bs[1]

#                 y_list.append((x, y, l, w))
            
#             x_list.append(y_list)
        
#         anchors_list.append(x_list)

#     return torch.from_numpy(np.array(anchors_list))


def generate_anchors(dimensions, box_sizes):
    """
    Function to generate anchors.
    :param dimensions:  Dimensions of extracted features
    :return:         A*i Anchors (A anchors per location i)
    """
    # print("Input dimensions:", dimensions)
    feat_h, feat_w = dimensions

    anchors_list = []

    for bs in box_sizes:
        x_list = []
        for i in range(0, feat_h):
            y_list = []
            for j in range(0, feat_w):
                x = (i - (bs[0] // 2))
                y = (j - (bs[1] // 2))
                # l = x + bs[0]
                l = bs[0]
                # w = y + bs[1]
                w = bs[1]

                y_list.append((x, y, l, w))

            x_list.append(y_list)

        anchors_list.append(x_list)

    return torch.from_numpy(np.array(anchors_list))


def calc_IOU(boxA, boxB):
    """
    Function taken from this site:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param boxA: (X, Y, Len, Wid)
    :param boxB: (X, Y, Len, Wid)
    :return: IOU area ratio.
    """
    boxA = boxA.float()
    boxB = boxB.float()

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # xB = min(boxA[2], boxB[2])
    # yB = min(boxA[3], boxB[3])
    xB = min(boxA[0] + boxA[2] - 1, boxB[0] + boxB[2] - 1)
    yB = min(boxA[1] + boxA[3] - 1, boxB[1] + boxB[3] - 1)

    # compute the area of intersection rectangle
    interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def calc_IOU2(boxA, boxB):
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

    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]

    # yB = boxA[1] + boxA[2]
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
