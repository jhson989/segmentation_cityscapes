import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from cityscapes.data_conf import labels

def get_mIoU(gt, pred):

    num_class = len(labels)-1
    gt = gt.cpu().detach().numpy().flatten()
    pred = torch.argmax(pred,dim=1)
    pred = pred.cpu().detach().numpy().flatten()

    current = confusion_matrix(gt, pred, labels=range(num_class))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection + 1e-6
    IoU = intersection / union.astype(np.float32)
    
    return np.mean(IoU)

    



