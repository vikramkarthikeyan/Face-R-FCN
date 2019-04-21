def generate_anchors(features, N, list_bb, box_size):
    """
    Function to generate Positive and Negative anchors.
    :param features:  Extracted features
    :param N:         Number of bounding boxes
    :param list_bb:   Bounding boxes (scaled to current configuration)
    :return:          Positive and negative anchors.
    """

    _, _, feat_h, feat_w = features.shape
    """Sizes for region proposals assuming batch input"""

    print("Shape of features:", features.shape)
    print("Shape of bounding box:", list_bb[0])

    # pos_anc = []
    # neg_anc = []
    # scale = 1

    anchors_list = []

    for bs in box_size:
        for i in range(0, feat_h):
            for j in range(0, feat_w):
                # max_iou = 0

                x = min(feat_h, max(0, (i - (bs[0] // 2))))
                y = min(feat_w, max(0, (j - (bs[1] // 2))))
                l = min(feat_h, x + bs[0]) - x
                w = min(feat_w, y + bs[1]) - y

                anchors_list.append((x, y, l, w))

    return anchors_list

    #             im_slice = features[x:x + bs[0], y:y + bs[1]]
    #             frame_a = (x, y, x + bs[0], y + bs[1])
    #             """
    #             Frame A is the sliding window for calculation.
    #             """
    #
    #             for bb in list_bb:
    #                 frame_b = (bb[1], bb[0], bb[1] + bb[3], bb[0] + bb[2])
    #                 """
    #                 Frame B is the input """
    #                 iou = calc_IOU(frame_a, frame_b)
    #                 if max_iou < iou:
    #                     max_iou = iou
    #
    #                 # print(max_iou)
    #
    #             if max_iou > 0.6:
    #                 print(frame_a, frame_b, max_iou)
    #                 # pos_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))
    #                 pos_anc.append([y * scale, x * scale,  bs[1], bs[0]])
    #             elif max_iou < 0.05:
    #                 # neg_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))
    #                 neg_anc.append([y * scale, x * scale,  bs[1], bs[0]])
    #
    # indices = np.random.permutation(len(neg_anc))
    #
    # neg_filtered = []
    # for idx in indices[:100]:
    #     neg_filtered.append(neg_anc[idx])
    #
    # return pos_anc, neg_filtered


def calc_IOU(boxA, boxB):
    """
    Function taken from this site:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    :param boxA: (X, Y, Len, Wid)
    :param boxB: (X, Y, Len, Wid)
    :return: IOU area ratio.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # xB = min(boxA[2], boxB[2])
    # yB = min(boxA[3], boxB[3])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

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
