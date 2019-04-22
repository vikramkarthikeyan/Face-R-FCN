PRE_NMS_TOP_N = 10000
POST_NMS_TOP_N = 2000
NMS_THRESH = 0.7
MIN_SIZE = 2
BATCHSIZE = 256

# IOU >= thresh: positive example
POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
NEGATIVE_OVERLAP = 0.3