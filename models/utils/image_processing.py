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
    batch_size = boxes.shape[0]
    results = []

    if scale_type == 'down':
        scale = 1/scale

    for batch_number in range(batch_size):
        batch = []
        for box in boxes[batch_number,:,:]:
            x = float(box[0]) * scale
            y = float(box[1]) * scale
            l = float(box[2]) * scale
            b = float(box[3]) * scale
            batch.append([x,y,l,b])
        results.append(batch)
    
    return np.array(results)