import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image

def plot_boxes(im, positive_anchors, negative_anchors, boxes):

    image = torchvision.transforms.ToPILImage()(im)
    im = np.array(image, dtype=np.uint8)

    print(im.shape)

    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    for i, box in enumerate(boxes):
        rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    
    if positive_anchors:
        for i, box in enumerate(positive_anchors):
            print(box)
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='green',facecolor='none')
            ax.add_patch(rect)
    
    if negative_anchors:
        for i, box in enumerate(negative_anchors):
            rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    plt.show()