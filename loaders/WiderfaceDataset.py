
import os
import torch
import scipy.io
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

# Ratios of anchors at each cell (width/height)
# A value of 1 represents a square anchor, and 0.5 is a wide anchor
RPN_ANCHOR_RATIOS = [0.5, 1, 2]


# https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html
class WiderFaceDataset(Dataset):

    def __init__(self, image_path, metadata_path, transform=None):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
        
        self.convert_to_image_list(self.metadata_path)
    
    def convert_to_image_list(self, path):
        self.f = scipy.io.loadmat(path)
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
        self.occlusion_label_list = self.f.get('occlusion_label_list')
        self.pose_label_list = self.f.get('pose_label_list')


        image_metadata = {
            'image_location': [],
            'image_ground_truth': []
        }

        for idx, event in enumerate(self.event_list):
            event = event[0][0]

            for file_idx, file_path in enumerate(self.file_list[idx][0]):
                file_name = file_path[0][0] + '.jpg'
                file_name = event + '/' +file_name
                file_path = os.path.abspath('data/widerface/WIDER_train/images/'+file_name)
                
                bounding_boxes = self.face_bbx_list[idx][0][file_idx][0]
                occlusions = self.occlusion_label_list[idx][0][file_idx][0]
                pose = self.pose_label_list[idx][0][file_idx][0]

                # Filter out medium and hard bounding_boxes if any 
                bounding_boxes_filtered = []
                for occlusion_idx, occlusion in enumerate(occlusions):
                    if occlusion[0] == 0:
                        bounding_boxes_filtered.append(bounding_boxes[occlusion_idx])

                if len(bounding_boxes_filtered) > 0:
                    image_metadata['image_location'].append(file_path)
                    image_metadata['image_ground_truth'].append(bounding_boxes_filtered)

                # self.plot_boxes(file_path,bounding_boxes_filtered)
                # self.generate_anchors(file_name)
                # break
            # break
        
        # convert to pandas
        self.dataset = pd.DataFrame.from_dict(image_metadata)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            with open(self.dataset.iloc[idx]['image_location'], 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')
                image, boxes = self.resize_image(image, self.dataset.iloc[idx]['image_ground_truth'])

        except Exception:
            print("Image not found..", traceback.format_exception())
            return ([], self.dataset.iloc[idx]['image_ground_truth'])

        image_tensor = self.pil2tensor(image)
        # ground_truth_tensor = boxes
        ground_truth_tensor = torch.tensor(np.array(boxes))

        return (image_tensor, ground_truth_tensor)
    
    # Referenced from: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    def resize_image(self, im, b_boxes, dimension=1024):
        old_size = im.size
        ratio = float(dimension)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)

        offset_x = (dimension-new_size[0])//2
        offset_y = (dimension-new_size[1])//2

        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (dimension, dimension))
        new_im.paste(im, (offset_x, offset_y))

        # Re-size and offset bounding boxes based on image
        results = []

        for box in b_boxes:
            x = int(abs(box[0]*ratio + offset_x))
            y = int(abs(box[1]*ratio + offset_y))
            l = int(box[2] * ratio)
            b = int(box[3] * ratio)
            new_box = [x, y, l, b]
            results.append(new_box)

        return new_im, results
        
    def generate_anchors(self, file_name):
        image = np.array(Image.open(file_name).convert('RGB'), dtype=np.uint32)
        print(generate_anchors(RPN_ANCHOR_SCALES[0], RPN_ANCHOR_RATIOS, image.shape, 4, 1))

    
    def plot_boxes(self, im, boxes):
        # im = np.array(Image.open(file).convert('RGB'), dtype=np.uint8)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Set colors for occlusions
        box_colors = {
            0: 'green',
            1: 'blue',
            2: 'red'
        }

        for i, box in enumerate(boxes):
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor=box_colors[0],facecolor='none')
            ax.add_patch(rect)

        plt.show()


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes
