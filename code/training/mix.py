import torch
import numpy as np


def rand_bbox(size, lam):
    """
    Retuns the coordinate of a random rectangle in the image for cutmix.

    Args:
        size (torch tensor [batch_size x c x W x H): Input size.
        lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.

    Returns:
        int: 4 coordinates of the rectangle.
        int: Proportion of the unmasked image.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies cutmix to a sample

    Args:
        x (torch tensor [batch_size x input_size]): Input batch.
        y (torch tensor [batch_size x num_classes]): Labels.
        alpha (float, optional): Parameter of the beta distribution. Defaults to 1..
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        torch tensor [batch_size x input_size]: Mixed input.
        torch tensor [batch_size x num_classes]: Mixed labels.
        float: Probability sampled by the beta distribution.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_y = y.clone()

    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_y[:, bbx1:bbx2, bby1:bby2] = y[index, bbx1:bbx2, bby1:bby2]

    return mixed_x, mixed_y
