from __future__ import absolute_import
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from .resnets import resnet50
from .utils.image_processing import scale_boxes_batch
from .utils.image_plotting import plot_boxes
from .config import rfcn_config, resnet_config
from .rpn import rpn
from .rpn import proposal_target_layer
from .psroi import psroi_pooling
from .psroi_pooling.modules.psroi_pool import PSRoIPool
    

class _RFCN(nn.Module):
    """ R-FCN """
    def __init__(self, num_classes=2):
        super(_RFCN, self).__init__()
        self.n_classes = num_classes

        # Initialize two types of losses
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # Define the RPN
        self.RCNN_rpn = rpn.RPN(rfcn_config.INPUT_CHANNELS_RPN, rfcn_config.ANCHOR_SIZES, rfcn_config.STRIDE, rfcn_config.IMAGE_VS_FEATURE_SCALE)

        # Define the proposal target layer
        self.RCNN_proposal_target = proposal_target_layer._ProposalTargetLayer(self.n_classes)

        # Define the pooling layers
        self.RCNN_psroi_pool_cls = PSRoIPool(resnet_config.POOLING_SIZE, resnet_config.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=resnet_config.POOLING_SIZE,
                                          output_dim=self.n_classes)
        self.RCNN_psroi_pool_loc = PSRoIPool(resnet_config.POOLING_SIZE, resnet_config.POOLING_SIZE,
                                          spatial_scale=1/16.0, group_size=resnet_config.POOLING_SIZE,
                                          output_dim=4)

        # Finally define the average pooling layer based on the grid size
        self.pooling = nn.AvgPool2d(kernel_size=resnet_config.POOLING_SIZE, stride=resnet_config.POOLING_SIZE)

    def forward(self, image, image_metadata, gt_boxes):

        # Add an extra dimension to the image tensor to handle batches in the future
        sizes = image[0].size()
        reshaped_image = image[0].reshape(1, sizes[0], sizes[1], sizes[2])
        
        # Pass the image onto the feature extractor
        base_features = self.RCNN_base(reshaped_image)

        # Calculate scale of features vs image 
        base_feature_dimension = base_features.shape[-1]
        image_dimension = reshaped_image.shape[-1]
        resize_scale = image_dimension / base_feature_dimension

        if rfcn_config.verbose:
            print("Image dimensions: {}. Feature dimensions: {}, scale: {}".format(reshaped_image.shape, base_features.shape, resize_scale))

        # Resize GT anchors to size of base features
        gt_boxes = scale_boxes_batch(gt_boxes, resize_scale, 'down')

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_features, image_metadata, gt_boxes)

        # if it is training phrase, then use ground truth bboxes for refining proposals
        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.RCNN_proposal_target(rois, gt_boxes, base_features)
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        
        if rfcn_config.verbose:
            print("ROIS generated:",rois.shape)
            print("\n\n----ROIS generated, moving onto PSROI----\n")

        rois = Variable(rois.cuda())
        base_features = Variable(base_features.cuda())

        base_features = self.RCNN_conv_new(base_features)

        # Get position based score maps
        cls_feat = self.RCNN_cls_base(base_features)
        bbox_base = self.RCNN_bbox_base(base_features)

        if rcfn_config.verbose:
            print("Features after conversion layer:",base_features.shape)
            print("PS Score maps for classification:", cls_feat.shape)
            print("PS Score maps for bounding boxes:", bbox_base.shape)

        # Flatten the ROIs generated from all the images in the batch as a single array of ROIs
        flattened_rois = rois.view(-1, 4)

        # Do PSROI average pooling on the position based score maps
        pooled_feat_cls = self.RCNN_psroi_pool_cls(cls_feat, flattened_rois)
        cls_score = self.pooling(pooled_feat_cls)
        cls_score = cls_score.squeeze()

        pooled_feat_loc = self.RCNN_psroi_pool_loc(bbox_base, flattened_rois)
        pooled_feat_loc = self.pooling(pooled_feat_loc)
        bbox_pred = pooled_feat_loc.squeeze()

        if rcfn_config.verbose:
            print("\n\n----PSROI----")
            print("\nAfter PSROI on score maps for classification:",pooled_feat_cls.shape)
            print("After PSROI on score maps for bounding boxes:",pooled_feat_loc.shape)
            print("\nAfter averaging score maps:", cls_score.shape)
            print("After averaging bbox_pred:", bbox_pred.shape)

        cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            RCNN_loss_cls, RCNN_loss_bbox = self.ohem_detect_loss(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        
        # Convert it to the batchwise format and return, TODO: Replace "1" with batch size hopefully soon
        cls_prob = cls_prob.view(1, rois.size(1), -1)
        bbox_pred = bbox_pred.view(1, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    
    def ohem_detect_loss(self, cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):

        

        def log_sum_exp(x):
            x_max = x.data.max()
            return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

        batch_size = 1

        num_hard = rfcn_config.PSROI_TRAINING_BATCH_SIZE * batch_size
        pos_idx = rois_label > 0
        num_pos = pos_idx.int().sum()

        if rfcn_config.verbose:
            print("\n\n-----OHEM-----")
            print("Number of positive examples:",num_pos.data)

        # classification loss
        num_classes = cls_score.size(1)
        weight = cls_score.data.new(num_classes).fill_(1.)
        weight[0] = num_pos.data / num_hard

        # Detach is used to clone a tensor which is removed from the computation graph
        cls_score_temp = cls_score.detach()
        rois_label_temp = rois_label.detach()

        cls_score_temp = cls_score_temp.cuda()
        rois_label_temp = rois_label_temp.cuda()

        # rank on cross_entropy loss
        loss_c = log_sum_exp(cls_score_temp) - cls_score_temp.gather(1, rois_label_temp.view(-1,1))
        loss_c[pos_idx] = 100. # include all positive samples
        _, topk_idx = torch.topk(loss_c.view(-1), num_hard)

        cls_score = cls_score.cuda()
        rois_label = rois_label.cuda()
        loss_cls = F.cross_entropy(cls_score[topk_idx], rois_label[topk_idx], weight=weight)
        
        # bounding box regression L1 loss
        pos_idx = pos_idx.unsqueeze(1).expand_as(bbox_pred)
        loc_p = bbox_pred[pos_idx].view(-1, 4)
        loc_t = rois_target[pos_idx].view(-1, 4)
        loss_box = F.smooth_l1_loss(loc_p.cuda(), loc_t.cuda())

        loss_cls = Variable(loss_cls, requires_grad=True)
        loss_box = Variable(loss_box, requires_grad=True)
        
        return loss_cls, loss_box


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        # self._init_weights()
