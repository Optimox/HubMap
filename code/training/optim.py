import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from training.lovasz import lovasz_loss

LOSSES = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]

def define_loss(name, weight=None, device="cuda"):
    """
    Defines the loss function associated to the name.
    Supports loss from torch.nn, see the LOSSES list.

    Args:
        name (str): Loss name.
        weight (list or None, optional): Weights for the loss. Defaults to None.
        device (str, optional): Device for torch. Defaults to "cuda".

    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch loss: Loss function
    """
    if weight is not None:
        weight = torch.FloatTensor(weight).to(device)
    if name in LOSSES:
        loss = getattr(torch.nn, name)(weight=weight, reduction="none")
    elif name == "lovasz":
        loss = lovasz_loss
    else:
        raise NotImplementedError

    return loss


def prepare_for_loss(y_pred, y_batch, loss, mode="mask", device="cuda", train=True):
    """
    Reformats predictions to fit a loss function.

    Args:
        y_pred (torch tensor): Predictions.
        y_batch (torch tensor): Truths.
        loss (str): Name of the loss function.
        mode (str, optional): Indicates what the model is predicting. Defaults to "mask".
        device (str, optional): Device for torch. Defaults to "cuda".
        train (bool, optional): Whether it is the training phase. Defaults to True.
    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch tensor: Reformated predictions
        torch tensor: Reformated truths
    """

    return y_pred.squeeze(), y_batch.squeeze()


def average_loss(loss, class_weights=None):
    """
    Handles averaging using class_weights

    Args:
        loss (torch tensor): Loss. Can be of size [BS x NUM_CLASSES] or [BS x NUM_CLASSES x H x W].
        class_weights (torch tensor [BS x NUM_CLASSES]): Class weights.

    Returns:
        torch tensor: Loss value.
    """
    if class_weights is None:
        return loss.mean()

    if len(loss.size()) == 4:  # pixel level
        loss = loss.mean(-1).mean(-1)

    assert len(loss.size()) == 2, "Wrong loss size, expected BS x NUM_CLASSES"

    
    return (loss * class_weights).mean()


def define_optimizer(name, params, lr=1e-3):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr)
    except AttributeError:
        raise NotImplementedError

    return optimizer
