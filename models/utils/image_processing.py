import numpy as np

def scale_boxes(boxes, scale, scale_type):
    results = []

    if scale_type == 'down':
        scale = 1/scale

    for box in boxes:
        x = int(float(box[0]) * scale)
        y = int(float(box[1]) * scale)
        l = int(float(box[2]) * scale)
        b = int(float(box[3]) * scale)
        results.append([x,y,l,b])
    
    return results

def scale_boxes_batch(boxes, scale, scale_type):
    
    if scale_type == 'down':
        scale = 1/scale
    
    boxes_temp = boxes.float()

    boxes_temp[:, :, 0] = boxes_temp[:, :, 0] * scale 
    boxes_temp[:, :, 1] = boxes_temp[:, :, 1] * scale
    boxes_temp[:, :, 2] = boxes_temp[:, :, 2] * scale
    boxes_temp[:, :, 3] = boxes_temp[:, :, 3] * scale

    return boxes_temp
