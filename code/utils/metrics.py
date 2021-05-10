import numpy as np


def dice_scores_img(pred, truth, eps=1e-8):
    """
    Dice metric for a single image as array.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    pred = pred.reshape(-1) > 0
    truth = truth.reshape(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return dice


def dice_scores_img_tensor(pred, truth, eps=1e-8):
    """
    Dice metric for a single image as tensor.

    Args:
        pred (torch tensor): Predictions.
        truth (torch tensor): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.

    Returns:
        np array : dice value for each class.
    """
    pred = pred.view(-1) > 0
    truth = truth.contiguous().view(-1) > 0
    intersect = (pred & truth).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    dice = (2.0 * intersect + eps) / (union + eps)
    return float(dice)


def dice_score(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric. Only classes that are present are weighted.

    Args:
        pred (np array): Predictions.
        truth (np array): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.

    Returns:
        float: dice value.
    """
    pred = (pred.reshape((truth.shape[0], -1)) > threshold).astype(int)
    truth = truth.reshape((truth.shape[0], -1)).astype(int)
    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


def dice_score_tensor(pred, truth, eps=1e-8, threshold=0.5):
    """
    Dice metric for tensors. Only classes that are present are weighted.

    Args:
        pred (torch tensor): Predictions.
        truth (torch tensor): Ground truths.
        eps (float, optional): epsilon to avoid dividing by 0. Defaults to 1e-8.
        threshold (float, optional): Threshold for predictions. Defaults to 0.5.

    Returns:
        float: dice value.
    """
    pred = (pred.view((truth.size(0), -1)) > threshold).int()
    truth = truth.view((truth.size(0), -1)).int()
    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)
    dice = (2.0 * intersect + eps) / (union + eps)
    return dice.mean()


def tweak_threshold(mask, pred):
    """
    Tweaks the threshold to maximise the score.

    Args:
        mask (torch tensor): Ground truths.
        pred (torch tensor): Predictions.

    Returns:
        float: Best threshold.
        float: Best score.
    """
    thresholds = []
    scores = []
    for threshold in np.linspace(0.2, 0.7, 11):

        dice_score = dice_scores_img_tensor(pred=pred > threshold, truth=mask)
        thresholds.append(threshold)
        scores.append(dice_score)

    return thresholds[np.argmax(scores)], np.max(scores)
