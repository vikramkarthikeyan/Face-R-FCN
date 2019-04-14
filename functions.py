def get_regions(features, N, list_bb):
    """
    Function to generate Positive and Negative anchors.
    :param features:  Extracted features
    :param N:         Number of bounding boxes
    :param list_bb:   Bounding boxes (scaled to current configuration)
    :return:          Positive and negative anchors.
    """

    box_size = [(3, 3), (6, 3), (3, 6)]
    """Sizes for region proposals"""

    feat_h, feat_w, _ = features.shape

    pos_anc = []
    neg_anc = []
    scale = 64

    for bs in box_size:
        for i in range(feat_h):
            for j in range(feat_w):

                max_iou = 0

                x = min(feat_h, max(0, (i - (bs[0] // 2))))
                y = min(feat_w, max(0, (j - (bs[1] // 2))))

                im_slice = features[x:x + bs[0], y:y + bs[1]]

                for bb in list_bb:
                    iou = calc_IOU((x, x + bs[0], y, y + bs[1]), bb)
                    if max_iou < iou:
                        max_iou = iou

                if max_iou > 0.8:
                    pos_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))
                elif max_iou < 0.2:
                    neg_anc.append((x * scale, (x + bs[0]) * scale, y * scale, (y + bs[1]) * scale))

    return pos_anc, neg_anc


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
