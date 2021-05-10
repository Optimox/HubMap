import torch
from training.lovasz import lovasz_loss

LOSSES = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]


def define_loss(name, device="cuda"):
    """
    Defines the loss function associated to the name.
    Supports losses in the LOSSES list, as well as the Lovasz, Softdice and Haussdorf losses.

    Args:
        name (str): Loss name.
        device (str, optional): Device for torch. Defaults to "cuda".

    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch loss: Loss function
    """
    if name in LOSSES:
        loss = getattr(torch.nn, name)(reduction="none")
    elif name == "lovasz":
        loss = lovasz_loss
    else:
        raise NotImplementedError

    return loss


def prepare_for_loss(y_pred, y_batch, loss, device="cuda", train=True):
    """
    Reformats predictions to fit a loss function.

    Args:
        y_pred (torch tensor): Predictions.
        y_batch (torch tensor): Truths.
        loss (str): Name of the loss function.
        device (str, optional): Device for torch. Defaults to "cuda".
        train (bool, optional): Whether it is the training phase. Defaults to True.
    Raises:
        NotImplementedError: Specified loss name is not supported.

    Returns:
        torch tensor: Reformated predictions
        torch tensor: Reformated truths
    """

    if loss in ["BCEWithLogitsLoss", "lovasz", "HaussdorfLoss", "SoftDiceLoss"]:
        y_batch = y_batch.to(device)
        y_pred = y_pred.squeeze(1)
        if not train:
            y_pred = y_pred.detach()
    else:
        raise NotImplementedError

    return y_pred, y_batch


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
