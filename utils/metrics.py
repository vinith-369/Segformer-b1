# ---------------------------------------------------------------
# Segmentation evaluation metrics: mIoU, pixel accuracy, per-class IoU
# ---------------------------------------------------------------

import numpy as np


def _fast_hist(label_true, label_pred, num_classes):
    """Compute confusion matrix for a pair of prediction/ground truth."""
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def scores(label_trues, label_preds, num_classes=19):
    """Compute mIoU, pixel accuracy, and per-class IoU.

    Args:
        label_trues: list of ground truth label arrays.
        label_preds: list of prediction label arrays.
        num_classes: number of classes (19 for Cityscapes).

    Returns:
        dict with 'Pixel Accuracy', 'Mean Accuracy', 'Mean IoU', 'Class IoU'.
    """
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0
    mean_iu = np.nanmean(iu[valid])

    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
