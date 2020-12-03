import numpy as np
import pandas as pd


def dice_scores_img(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric for a single images.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.

    Returns:
        np array : dice value for each class
    """
    pred = pred.reshape(-1)>0
    truth = truth.reshape(-1)>0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return dice


def dice_score(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric. Only classes that are present are weighted.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.

    Returns:
        float: dice value
    """
    pred = (pred.reshape((truth.shape[0], -1)) > threshold).astype(int)
    truth = truth.reshape((truth.shape[0], -1)).astype(int)
    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def update(self, y_batch, preds):
        self.y_mask.append(y_batch.cpu().numpy())

        pred_mask = preds.cpu().numpy()

        self.pred_mask.append(pred_mask)

    def concat(self):
        self.pred_mask = np.concatenate(self.pred_mask)
        self.y_mask = np.concatenate(self.y_mask)

    def compute(self):
        self.concat()
        self.metrics["dice"].append(dice_score(self.pred_mask, self.y_mask))

        return self.metrics

    def reset(self):
        self.pred_mask = []
        self.y_mask = []

        self.metrics = {
            "dice": [],
        }
