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