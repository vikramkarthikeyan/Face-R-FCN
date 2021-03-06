import numpy as np
import torch


def generate_anchors(dimensions, box_sizes):
    """
    Function to generate anchors.
    :param dimensions:  Dimensions of extracted features
    :return:         A*i Anchors (A anchors per location i)
    """
    feat_h, feat_w = dimensions

    anchors_list = []

    for bs in box_sizes:
        x_list = []
        for i in range(0, feat_h):
            y_list = []
            for j in range(0, feat_w):
                x = (i - (bs[0] // 2))
                y = (j - (bs[1] // 2))
                l = bs[0]
                w = bs[1]

                y_list.append((x, y, l, w))

            x_list.append(y_list)

        anchors_list.append(x_list)

    return np.array(anchors_list, dtype=np.float)


def calc_IOU_vectorized_(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)

    x12 = x11 + x12 - 1
    y12 = y11 + y12 - 1

    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

    x22 = x21 + x22 - 1
    y22 = y21 + y22 - 1
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    
    return iou


def calc_IOU_vectorized_tmp(bboxes1, bboxes2):

    x11, y11, x12, y12 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]

    x12 = x11 + x12 - 1
    y12 = y11 + y12 - 1

    x21, y21, x22, y22 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x22 = x21 + x22 - 1
    y22 = y21 + y22 - 1

    # determine the (x, y)-coordinates of the intersection rectangle
    #xA = np.maximum(np.expand_dims(x21, axis=1), np.expand_dims(x11, axis=0))
    xA = np.maximum(x21, np.expand_dims(x11, axis=0).T)
    yA = np.maximum(y21, np.expand_dims(y11, axis=0).T)
    xB = np.maximum(x22, np.expand_dims(x12, axis=0).T)
    yB = np.maximum(y22, np.expand_dims(y12, axis=0).T)
    # yA = torch.max(y21, torch.t(y11.view(1, -1)))
    # xB = torch.min(x22, torch.t(x12.view(1, -1)))
    # yB = torch.min(y22, torch.t(y12.view(1, -1)))

    print("XA:", xA.shape)

    print("YA:", yA.shape)
    # label_zero = torch.tensor(0.0)

    # compute the area of intersection rectangle
    # interArea = torch.max((xB - xA + 1), label_zero) * torch.max((yB - yA + 1), label_zero)
    interArea = np.max((xB - xA + 1), 0) * np.max((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    boxBArea = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)

    print("boxAArea:", boxAArea.shape)
    print("boxBArea:", boxBArea.shape)
    print("interArea:", interArea.shape)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxBArea + boxAArea.T - interArea)

    return torch.abs(iou)


def calc_IOU_vectorized(bboxes1, bboxes2):
    x11, y11, x12, y12 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]

    x12 = x11 + x12 - 1
    y12 = y11 + y12 - 1

    x21, y21, x22, y22 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x22 = x21 + x22 - 1
    y22 = y21 + y22 - 1

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(x21, torch.t(x11.view(1, -1)))
    yA = torch.max(y21, torch.t(y11.view(1, -1)))
    xB = torch.min(x22, torch.t(x12.view(1, -1)))
    yB = torch.min(y22, torch.t(y12.view(1, -1)))

    label_zero = torch.tensor(0.0)

    # compute the area of intersection rectangle
    interArea = torch.max((xB - xA + 1), label_zero) * torch.max((yB - yA + 1), label_zero)

    boxAArea = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    boxBArea = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxBArea + torch.t(boxAArea.view(1, -1)) - interArea)

    return torch.abs(iou)


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
