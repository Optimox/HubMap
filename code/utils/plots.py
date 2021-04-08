import cv2
import numpy as np
import plotly.express as px

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
    cv2.polylines(img, contours, True, (0.7, 0.0, 0.0), w)

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


def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_OCEAN):
    """
    Overlays an heatmat with an image.

    Args:
        heatmap (numpy array): Attention map.
        image (numpy array): Image.
        alpha (float, optional): Transparency. Defaults to 0.5.
        colormap (cv2 colormap, optional): Colormap. Defaults to cv2.COLORMAP_OCEAN.

    Returns:
        numpy array: image with the colormap.
    """
    if np.max(image) <= 1:
        image = (image * 255).astype(np.uint8)

    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap[0, 0] = 255

    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image[:, :, [2, 1, 0]], alpha, heatmap, 1 - alpha, 0)

    return output[:, :, [2, 1, 0]]


def plot_contours_preds(img, mask, preds, w=1, downsize=1):
    """
    Plots the contours of a given mask.

    Args:
        img (numpy array [H x W x C]): Image.
        mask (numpy array [H x W]): Mask.
        w (int, optional): Contour width. Defaults to 1.

    Returns:
        img (numpy array [H x W x C]): Image with contours.
    """
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)
    if mask.max() > 1:
        mask = (mask / 255).astype(float)
    mask = (mask * 255).astype(np.uint8)
    if preds.max() > 1:
        preds = (preds / 255).astype(float)
    preds = (preds * 255).astype(np.uint8)

    if downsize > 1:
        new_shape = (mask.shape[1] // downsize, mask.shape[0] // downsize)
        mask = cv2.resize(
            mask,
            new_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        img = cv2.resize(
            img,
            new_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        preds = cv2.resize(
            preds,
            new_shape,
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_preds, _ = cv2.findContours(preds, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img_pred = img.copy()
    cv2.polylines(img, contours, True, (1.0, 0.0, 0.0), w)
    cv2.polylines(img_pred, contours_preds, True, (0.0, 1.0, 0.0), w)

    img = (img + img_pred) / 2

    return px.imshow(img)


def plot_heatmap_preds(img, mask, preds, w=1, downsize=1):
    """
    Plots the contours of a given mask.

    Args:
        img (numpy array [H x W x C]): Image.
        mask (numpy array [H x W]): Mask.
        w (int, optional): Contour width. Defaults to 1.

    Returns:
        img (numpy array [H x W x C]): Image with contours.
    """
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)
    if mask.max() > 1:
        mask = (mask / 255).astype(float)
    mask = (mask * 255).astype(np.uint8)

    if downsize > 1:
        new_shape = (mask.shape[1] // downsize, mask.shape[0] // downsize)
        mask = cv2.resize(
            mask,
            new_shape,
            interpolation=cv2.INTER_NEAREST,
        )
        img = cv2.resize(
            img,
            new_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        preds = cv2.resize(
            preds,
            new_shape,
            interpolation=cv2.INTER_LINEAR,
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(img, contours, True, (1.0, 0.0, 0.0), w)

    heatmap = 1 - preds / 2
    heatmap[0, 0] = 1
    img = overlay_heatmap(heatmap, img.copy(), alpha=0.7, colormap=cv2.COLORMAP_HOT)

    return px.imshow(img)
