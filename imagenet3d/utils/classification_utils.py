import numpy as np


def eval_classification(cls_gt, cls_pred):
    return (cls_gt == cls_pred).astype(np.int32)
