import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torchsummary import summary
import EarlyStopping
import AverageMeter
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def custom_collate(batch):

    #(images, ground_truth_boxes)
    images = [item[0] for item in batch]
    ground_truth_boxes = [item[1] for item in batch]
    image_paths = [item[2] for item in batch]
    return [images, ground_truth_boxes, image_paths]



# Followed PyTorch's ImageNet documentation
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Trainer:

    def __init__(self, training_data, validation_data, num_classes=2, training_batch_size=5, validation_batch_size=5): 

        # Create training dataloader
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=training_batch_size, shuffle=False,
                                                             num_workers=1, collate_fn=custom_collate)

        # Create validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_batch_size, shuffle=False,
                                                             collate_fn=custom_collate, num_workers=5)

        self.num_classes = num_classes
        self.early_stopper = EarlyStopping.EarlyStopper()

    
    def train(self, model, criterion, optimizer, epoch, usegpu):
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()
        top5 = AverageMeter.AverageMeter()

        # switch to train mode
        model.train()

        loss_temp = 0

        start = time.time()

        torch.cuda.empty_cache()

        for i, (images, targets, image_paths) in enumerate(self.train_loader):

            for j in range(len(targets)):

                print("\n\n\n\n................ Batch run ................\n\n\n\n")

                data, target = Variable(images[j]), Variable(targets[j], requires_grad=False)

                if usegpu:
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                
                target = np.array(target)
                target = np.reshape(target, (1, target.shape[0], target.shape[1]))

                model.zero_grad()

                # Compute Model output and loss
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = model([data], [image_paths[j]], target)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean()
                #    + RCNN_loss_bbox.mean()
                
                print(rpn_loss_cls.mean())
                print(rpn_loss_box.mean())
                print(RCNN_loss_cls.mean())
                print(RCNN_loss_bbox.mean())
                print(loss)


                loss_temp = 0
                start = time.time()

                # Clear(zero) Gradients for theta
                optimizer.zero_grad()

                # Perform BackProp wrt theta
                loss.backward()

                # Update theta
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - start)
                start = time.time()

                # print('\rTraining - Epoch [{:04d}] Batch [{:04d}/{:04d}]\t'
                #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                #             epoch, i, len(self.train_loader), batch_time=batch_time,
                #             loss=losses), end="")
            
        print("\nTraining Accuracy: Acc@1: {top1.avg:.3f}%, Acc@5: {top5.avg:.3f}%".format(top1=top1, top5=top5))

    def normalize(self, features):
        norm = np.linalg.norm(features)
        if norm == 0: 
            return features
        return features/norm

    def validate(self, model, epoch, usegpu):

        # switch to evaluate mode
        model.eval()

        print("\nRunning Verification Protocol")

        similarity_scores = []
        actual_scores = []

        

        with torch.no_grad():
            for i, (template_1, template_2, subject_1, subject_2, template_n1, template_n2) in enumerate(self.validation_loader):
                
                for j in range(len(template_1)):
            
                    template_left = Variable(template_1[j])
                    template_right = Variable(template_2[j])

                    if usegpu:
                        template_left = template_left.cuda(non_blocking=True)
                        template_right = template_right.cuda(non_blocking=True)

                    outputs = []

                    def hook(module, input, output):
                        outputs.append(output)

                    model.avgpool.register_forward_hook(hook)

                    # Compute outputs of two templates
                    output_1 = model(template_left)
                    output_1 = outputs[0]

                    outputs = []

                    output_2 = model(template_right)
                    output_2 = outputs[0]

                    # Compute average of all the feature vectors into a single feature vector
                    output_1 = np.average(output_1.cpu().numpy(), axis=0).flatten().reshape(1, -1)
                    output_2 = np.average(output_2.cpu().numpy(), axis=0).flatten().reshape(1, -1)

                    # Compute Cosine Similarity with normalization in the formula
                    similarity = cosine_similarity(output_1, output_2)
                    similarity = similarity[0][0]

                    similarity_scores.append(similarity)

                    if subject_1[j] == subject_2[j]:
                        actual_scores.append(1)
                    else:
                        actual_scores.append(0)

                print("\rPair Batch({:05d}/{:05d})".format(i, len(self.validation_loader)), end="")

        return similarity_scores, actual_scores


    def save_checkpoint(self, state, filename='./models/checkpoint.pth.tar'):
        torch.save(state, filename)
    
    # Used - https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
