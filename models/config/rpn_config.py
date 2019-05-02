PRE_NMS_TOP_N = 2000
POST_NMS_TOP_N = 200

NMS_THRESH = 0.9

MIN_SIZE = 2
BATCHSIZE = 256


# IOU >= thresh: positive example
POSITIVE_OVERLAP = 0.9
# IOU < thresh: negative example
NEGATIVE_OVERLAP = 0.3