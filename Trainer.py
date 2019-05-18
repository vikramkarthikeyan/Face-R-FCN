import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import pandas as pd

from torch.autograd import Variable

from  models.utils.image_processing import scale_boxes_batch

import torchvision
import EarlyStopping
import AverageMeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import models.Hook as Hook
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

    def __init__(self, training_data, validation_data, num_classes=2, training_batch_size=1, validation_batch_size=1):

        # Create training dataloader
        self.train_loader = torch.utils.data.DataLoader(training_data, batch_size=training_batch_size, shuffle=True,
                                                        num_workers=0, collate_fn=custom_collate)

        # Create validation dataloader
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=validation_batch_size,
                                                             shuffle=False,
                                                             collate_fn=custom_collate, num_workers=0)

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
                data, target = Variable(images[j], requires_grad=True), Variable(targets[j], requires_grad=False)

                data = data.cuda(non_blocking=True)
                target = np.array(target, dtype=np.float)
                target = np.expand_dims(target, axis=0)

                try:
                    model.zero_grad()

                    # Compute Model output and loss
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = model([data], [image_paths[j]], target)

                    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                    file_ip.append([i, loss.item(), rpn_loss_cls.mean().item(), rpn_loss_box.mean().item(),
                                    RCNN_loss_cls.mean().item(), RCNN_loss_bbox.mean().item()])

                    if i % 200 == 0:
                        print(
                            "Memory allocated tensors: {}, cached: {} ".format(torch.cuda.memory_allocated(device=None),
                                                                               torch.cuda.memory_cached(device=None)))
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

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print("out of memory...clearing cache...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        break

            if i == 1:
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

                    #print(cls_prob, bbox_pred, rois)

                    rois = rois[0,:,:].cpu().numpy()
                    bbox_pred = bbox_pred[0,:,:].cpu().numpy()

                    # apply bbox transformations
                    rois = bbox_transform(rois, bbox_pred)

                    # Perform NMS on ROIs
                    keep_rois_postNMS = nms_numpy(rois, 0.7)
                    rois = rois[keep_rois_postNMS, :]
                    cls_prob = cls_prob[0, keep_rois_postNMS, :]

                    # get those ROIs with higher face prob
                    keep = cls_prob[:, 1] > 0.5

                    rois = rois[keep, :]

                    rois = np.expand_dims(rois, 0)
                    
                    # Write logic for comparing GT_boxes and ROIs
                    plot_boxes(data, scale_boxes_batch(rois, 16, "up"), targets, str(i))

                print("Image {}/{}".format(i, len(self.validation_loader)))

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


def plot_boxes(image, rois, gt_boxes,  image_count):
    image = torchvision.transforms.ToPILImage()(image.cpu())
    im = np.array(image, dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # if positive_anchors:
    for i, box in enumerate(rois[0]):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    for i, box in enumerate(gt_boxes[0]):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    # plt.show()
    plt.savefig('validation_plots/regions_{}.png'.format(image_count))

def bbox_transform(boxes, split_deltas):
    results =  np.full(boxes.shape, 0.0, dtype=np.float)
    
    results[:,0] = (split_deltas[:,0] * boxes[:,0]) + boxes[:,0]
    results[:,1] = (split_deltas[:,1] * boxes[:,1]) + boxes[:,1]
    results[:,2] = np.exp(split_deltas[:,2]) * boxes[:,2]
    results[:,3] = np.exp(split_deltas[:,3]) * boxes[:,3]
    return results
    

def nms_numpy(entries, thresh):
    x1 = entries[:, 0]
    y1 = entries[:, 1]
    l = entries[:, 2]
    b = entries[:, 3]
    # COMMENTED FOR SPEED
    # TODO: check why it is needed
    # scores = entries[:, 4]

    x2 = x1 + l - 1
    y2 = y1 + b - 1

    # Initialize list of picked indices
    idx_keep = []

    # Calculate areas of all bounding boxes
    areas = l * b

    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    # _, idxs = torch.sort(y2)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while idxs.shape[0] > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        ind = idxs[-1]
        idx_keep.append(ind)

        XX1 = np.clip(x1[idxs], x1[ind], None)
        YY1 = np.clip(y1[idxs], y1[ind], None)

        XX2 = np.clip(x2[idxs], None, x2[ind])
        YY2 = np.clip(y2[idxs], None, y2[ind])

        W = np.clip((XX2 - XX1 + 1), 0, None)
        H = np.clip((YY2 - YY1 + 1), 0, None)

        # mask = ((W * H).float() / areas.float()).lt(thresh)
        mask = ((W * H) / areas) < thresh
        mask[-1] = False

        areas = areas[mask]
        idxs = idxs[mask]

    # return only the bounding boxes that were picked
    return torch.from_numpy(np.array(idx_keep))
