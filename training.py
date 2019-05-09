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
training_image_path = os.path.abspath("data/widerface/WIDER_train/images")
training_metadata_path = os.path.abspath("data/widerface/wider_face_split/wider_face_train.mat")

validation_image_path = os.path.abspath("data/widerface/WIDER_val/images")
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

    # -------------------------------------------------
    print("\nChecking if a GPU is available...")
    use_gpu = torch.cuda.is_available()

    # Initialize new model
    if use_gpu:
        model = model.cuda()
        print ("Using GPU")
    else:
        print ("GPU is unavailable")
        exit() 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=SGD_MOMENTUM)

    start_epochs = 0 
    model_name = './saved_models/checkpoint.pth.tar'

    # Get checkpoint if available
    if args.retrain and os.path.isfile(model_name):

        print("Found already trained model for this split...")
        
        if use_gpu:
            checkpoint = torch.load(model_name)
        else:
            checkpoint = torch.load(model_name, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epochs = checkpoint['epoch']
    
    end_epochs = start_epochs + EPOCHS

    highest_accuracy = 0
    highest_accuracy_5 = 0

    print("\nInitiating training...\n")

    for epoch in range(start_epochs, end_epochs):
    
        # Train for one Epoch
        trainer.train(model, criterion, optimizer, epoch, use_gpu)

        # Checkpointing the model after every epoch
        trainer.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_accuracy': 0,
                        'optimizer' : optimizer.state_dict(),
        }, model_name)








