import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,Input

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

@tf.function
def to_box_cords(bboxs):
    X = bboxs[:,0]
    Y = bboxs[:,1]
    W = bboxs[:,2]
    H = bboxs[:,3]
    X_1 = X - (W/2)
    Y_1 = Y - (H/2)
    X_2 = X + (W/2)
    Y_2 = Y + (H/2)
    return tf.stack([Y_1,X_1,Y_2,X_2],axis=1)

def IoU(dataset, model, target_size,IoUThreshold=0.7):
    IoUTotal = 0
    count = 0
    correct = 0
    IoUs = []

    for IMG, Y in dataset:
        Y_hat = model(IMG)
        for i in range(Y.shape[0]):
            y = Y[i]
            y_hat = Y_hat[i]
            box_true = to_box_cords(Y).numpy()
            box_pred = to_box_cords(Y_hat).numpy()

            boxA = box_true[i] * target_size[0]
            boxB = box_pred[i] * target_size[0]

            iou = bb_intersection_over_union(boxA,boxB)

            count += 1
            IoUTotal += iou
            IoUs.append(iou)

            if iou >= IoUThreshold:
                correct += 1
    return correct, correct/count, IoUs