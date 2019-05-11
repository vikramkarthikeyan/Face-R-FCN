import pandas as pd
import argparse
import torch.nn as nn
import torch
import os
import numpy as np
import imp

from torchvision import models
from tqdm import tqdm
from loaders import WiderfaceDataset
from Trainer import Trainer
from models.resnets import RFCN_resnet

parser = argparse.ArgumentParser()
parser.add_argument('--retrain', dest='retrain', action='store_true', default=False)
args = parser.parse_args()

# Paths
training_image_path = os.path.abspath("data/widerface/WIDER_train/images/")
training_metadata_path = os.path.abspath("data/widerface/wider_face_split/wider_face_train.mat")

validation_image_path = os.path.abspath("data/widerface/WIDER_val/images/")
validation_metadata_path = os.path.abspath("data/widerface/wider_face_split/wider_face_val.mat")


# Hyperparameters
LR = 0.01
SGD_MOMENTUM = 0.9
EPOCHS = 50

if __name__ == "__main__":

    # Create WIDERFACE dataset object for training
    training_set = WiderfaceDataset.WiderFaceDataset(training_image_path, training_metadata_path)

    # Create WIDERFACE dataset object for validation
    validation_set = WiderfaceDataset.WiderFaceDataset(validation_image_path, validation_metadata_path)

    # Initialize trainer
    trainer = Trainer(training_set, validation_set)

    # Initialize R-FCN model
    model = RFCN_resnet(pretrained=True)
    model.create_architecture()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=SGD_MOMENTUM)

    model_name = './saved_models/2_epoch_model.pth.tar'    
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("\nInitiating Inference...\n")

    # Validate the model
    trainer.validate(model, 0)









