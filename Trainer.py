import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import pandas as pd

from torch.autograd import Variable

import torchvision
import EarlyStopping
import AverageMeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image



def custom_collate(batch):
    # (images, ground_truth_boxes)
    images = [item[0] for item in batch]
    ground_truth_boxes = [item[1] for item in batch]
    image_paths = [item[2] for item in batch]
    return [images, ground_truth_boxes, image_paths]


# Followed PyTorch's ImageNet documentation
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class Trainer:

    def __init__(self, training_data, validation_data, num_classes=2, training_batch_size=5, validation_batch_size=5):

        # Create training dataloader
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=training_batch_size, shuffle=True,
                                                        num_workers=5, collate_fn=custom_collate)

        # Create validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_batch_size,
                                                             shuffle=False,
                                                             collate_fn=custom_collate, num_workers=5)

        self.num_classes = num_classes
        self.early_stopper = EarlyStopping.EarlyStopper()

    def train(self, model, criterion, optimizer, epoch):
        batch_time = AverageMeter.AverageMeter()
        losses = AverageMeter.AverageMeter()
        top1 = AverageMeter.AverageMeter()
        top5 = AverageMeter.AverageMeter()

        file_ip = []

        # switch to train mode
        model.train()

        start = time.time()

        torch.cuda.empty_cache()

        for i, (images, targets, image_paths) in enumerate(self.train_loader):

            start = time.time()

            for j in range(len(targets)):
                data, target = Variable(images[j]), Variable(targets[j], requires_grad=False)

                data = data.cuda(non_blocking=True)
                target = np.array(target, dtype=np.float)
                target = np.expand_dims(target, axis=0)

                model.zero_grad()

                # Compute Model output and loss
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = model([data], [image_paths[j]], target)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                file_ip.append([i, loss.item(), rpn_loss_cls.mean().item(), rpn_loss_box.mean().item(),
                                RCNN_loss_cls.mean().item(), RCNN_loss_bbox.mean().item()])

                if i % 50 == 0:
                    pd.DataFrame(data=file_ip,
                                 columns=["Batch", "Loss", "RPN Classification Loss", "RPN Regression Loss",
                                          "RCNN Classification Loss", "RCNN Regression Loss"]
                                 ).to_csv(path_or_buf="losses.csv")

                # Clear(zero) Gradients for theta
                optimizer.zero_grad()

                # Perform BackProp wrt theta
                loss.backward()

                # Update theta
                optimizer.step()

                losses.update(loss)

                if i == 200:
                    break

            # measure elapsed time
            batch_time.update(time.time() - start)

            print('\rTraining - Epoch [{:04d}] Batch [{:04d}/{:04d}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(self.train_loader),
                                                                  batch_time=batch_time, loss=losses), end="")

        print("\nTraining Accuracy: Acc@1: {top1.avg:.3f}%, Acc@5: {top5.avg:.3f}%".format(top1=top1, top5=top5))

        pd.DataFrame(data=file_ip, columns=["Batch", "Loss", "RPN Classification Loss", "RPN Regression Loss",
                                            "RCNN Classification Loss", "RCNN Regression Loss"]
                     ).to_csv(path_or_buf="losses.csv")

    def save_checkpoint(self, state, filename='./saved_models/checkpoint.pth.tar'):
        torch.save(state, filename)

    def validate(self, model, epoch):

        # Inference mode
        model.eval()

        with torch.no_grad():

            for i, (images, targets, image_paths) in enumerate(self.validation_loader):

                start = time.time()

                for j in range(len(targets)):
                    data, target = Variable(images[j]), Variable(targets[j], requires_grad=False)
                    data = data.cuda(non_blocking=True)
                    target = torch.tensor(target)
                    target = target.unsqueeze(0)

                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = model([data], [image_paths[j]], target)

                    print(cls_prob.shape, bbox_pred.shape)
                    print(cls_prob, bbox_pred)
                    # Write logic for comparing GT_boxes and ROIs
                    image_location = image_paths[j]
                    #plot_boxes(image_location, bbox_pred, [], [])

                break



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


def plot_boxes(file, positive_anchors, negative_anchors, boxes):

    im = np.array(Image.open(file).convert('RGB'), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    
    print("acnhors generatd!:",positive_anchors.shape)
    #for i, box in enumerate(boxes):
       # rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='blue', facecolor='none')
       # ax.add_patch(rect)
    
    #if positive_anchors:
    for i, box in enumerate(positive_anchors):
            print(box)
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='green',facecolor='none')
            ax.add_patch(rect)
    
    if negative_anchors:
        for i, box in enumerate(negative_anchors):
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    plt.show()
    plt.savefig('regions.png')
