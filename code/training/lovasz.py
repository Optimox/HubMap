# Adapted from https://github.com/bermanmaxim/LovaszSoftmax

import torch
import torch.nn.functional as F

from torch.autograd import Variable


def flatten(scores, labels):
    """
    Flattens predictions in the batch (binary case)
    """
    return scores.view(-1), labels.view(-1)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -infty and +infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)

    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
    """
    if per_image:
        loss = torch.stack([lovasz_hinge_flat(
            *flatten(log.unsqueeze(0), lab.unsqueeze(0))
        ) for log, lab in zip(logits, labels)])

    else:
        loss = lovasz_hinge_flat(*flatten(logits, labels))
    return loss


def symmetric_lovasz(outputs, targets):
    targets = targets.float()
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2


def lovasz_loss(x, y):
    """
    Computes the symetric lovasz for each class.

    Args:
        x (torch tensor [BS x H x W]): Logits.
        y (torch tensor [BS x H x W]): Ground truth.

    Returns:
        torch tensor [BS]: Loss values.
    """
    return symmetric_lovasz(x, y)
