import cv2
import numpy as np
import plotly.express as px


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


def plot_contours_preds(img, preds, mask=None, w=1, downsize=1):
    """
    Plots the contours of mask predictions (in green) and of a mask (in red).

    Args:
        img (numpy array [H x W x C]): Image.
        preds (numpy int array [H x W] or None): Predicted mask.
        mask (numpy array [H x W] or None): Mask.
        w (int, optional): Contour width. Defaults to 1.
        downsize (int, optional): Downsizing factor. Defaults to 1.

    Returns:
        px.imshow: Ploty plot.
    """
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)
    if mask is not None:
        if mask.max() > 1:
            mask = (mask / 255).astype(float)
        mask = (mask * 255).astype(np.uint8)

    if preds.max() > 1:
        preds = (preds / 255).astype(float)
    preds = (preds * 255).astype(np.uint8)

    if downsize > 1:
        new_shape = (preds.shape[1] // downsize, preds.shape[0] // downsize)

        if mask is not None:
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

    contours_preds, _ = cv2.findContours(preds, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(img, contours_preds, True, (0.0, 1.0, 0.0), w)

    if mask is not None:
        img_gt = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img_gt, contours, True, (1.0, 0.0, 0.0), w)
        img = (img + img_gt) / 2

    return px.imshow(img)


def plot_heatmap_preds(img, preds, mask=None, w=1, downsize=1):
    """
    Plots the heatmap of predictions and the contours of a mask (in red).

    Args:
        img (numpy array [H x W x 3]): Image.
        preds (numpy float array [H x W] or None): Predicted probabilities.
        mask (numpy array [H x W] or None): Mask.
        w (int, optional): Contour width. Defaults to 1.
        downsize (int, optional): Downsizing factor. Defaults to 1.

    Returns:
        px.imshow: Ploty plot.
    """
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)
    if mask is not None:
        if mask.max() > 1:
            mask = (mask / 255).astype(float)
        mask = (mask * 255).astype(np.uint8)

    if downsize > 1:
        new_shape = (preds.shape[1] // downsize, preds.shape[0] // downsize)

        if mask is not None:
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

    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(img, contours, True, (1.0, 0.0, 0.0), w)

    heatmap = 1 - preds / 2
    heatmap[0, 0] = 1
    img = overlay_heatmap(heatmap, img.copy(), alpha=0.7, colormap=cv2.COLORMAP_HOT)

    return px.imshow(img)
