import pandas as pd
import argparse
import torch.nn as nn
import torch
import os
import numpy as np
import imp

from torchvision import models
from tqdm import tqdm
from torchsummary import summary
from loaders import WiderfaceDataset

parser = argparse.ArgumentParser()

parser.add_argument("--split", help="The split number in the dataset",
                    type=int)

parser.add_argument('--retrain', dest='retrain', action='store_true', default=False)
 
args = parser.parse_args()

# Paths
image_path = os.path.abspath("data/widerface/WIDER_train/images")
metadata_path = os.path.abspath("data/widerface/wider_face_split/wider_face_train.mat")


# Hyperparameters
LR = 0.01
SGD_MOMENTUM = 0.9
EPOCHS = 50

if __name__ == "__main__":

    # Create IJBA dataset object for training
    training_set = WiderfaceDataset.WiderFaceDataset(image_path, metadata_path)



