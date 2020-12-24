import cv2
import numpy as np

from matplotlib import pyplot as plt

from utils.metrics import dice_scores_img


def plot_contours(img, mask, w=1):
    """
    Plots the contours of a given mask.

    Args:
        img (numpy array [H x W x C]): Image.
        mask (numpy array [H x W]): Mask.
        w (int, optional): Contour width. Defaults to 1.

    Returns:
        img (numpy array [H x W x C]): Image with contours.
    """
    if img.max() > 1:
        img = (img / 255).astype(float)
    if mask.max() > 1:
        mask = (mask / 255).astype(float)
    mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(img, contours, True, (0.7, 0., 0.), w)

    plt.imshow(img)

    return img


def plot_global_pred(mask, pred):
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].imshow(mask)
    axs[0].set_title("Labels")

    axs[1].imshow(pred)
    axs[1].set_title("Predictions")
    plt.show()


def plot_thresh_scores(mask, pred, plot=True):
    thresholds = []
    scores = []
    for threshold in np.linspace(0.2, 0.7, 11):
        dice_score = dice_scores_img(pred=pred > threshold, truth=mask)
        thresholds.append(threshold)
        scores.append(dice_score)

    if plot:
        plt.plot(thresholds, scores)
        plt.show()

    return thresholds[np.argmax(scores)], np.max(scores)
