INPUT_CHANNELS_RPN = 1024
ANCHOR_SIZES = [
    (1, 1), (2, 1), (1, 2), (2, 2), 
    (3, 3), (2, 3), (3, 2), (2, 4), 
    (4, 2), (4, 4),  (5, 5), (5, 6), 
    (6, 5), (4, 8), (8, 4), (8, 8),
    (8, 16), (16, 8), (16,16), (32,32)]

# ANCHOR_SIZES = [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (4, 2), (4,4), (4, 8), (8, 4), (8,8),
#                 (8, 16), (16, 8), (16,16), (32, 16), (16, 32), (32,32), (32, 64), (64, 32)]

NUM_ANCHORS = len(ANCHOR_SIZES)
STRIDE = 1
IMAGE_VS_FEATURE_SCALE = 8


OHEM = True
ROI_BATCH_SIZE = 50
FG_FRACTION = 0.33
PSROI_TRAINING_BATCH_SIZE = 32


IMAGE_INPUT_DIMS = 1024


################################################
# Config for Proposal Target Layer

# Overlap threshold for a ROI to be considered Face (if >= FACE_THRESH)
FACE_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.1

RPN_NEGATIVE_OVERLAP = 0.05
RPN_POSITIVE_OVERLAP = 0.5

UNIFORM_EXAMPLE_WEIGHTING = True
RPN_L1_DELTA = 1.0

RPN_MAX_BG = 200

BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
