from matplotlib import pyplot as plt
from utils.metrics import dice_scores_img
import numpy as np


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
    for threshold in np.linspace(0.3, 0.7, 5):
        dice_score = dice_scores_img(pred=pred > threshold, truth=mask)
        thresholds.append(threshold)
        scores.append(dice_score)

    if plot:
        plt.plot(thresholds, scores)
        plt.show()

    return thresholds[np.argmax(scores)], np.max(scores)
