import torch
import numpy as np
import torch.nn as nn

from segmentation_models_pytorch.encoders.efficientnet import EfficientNetEncoder


class MixStyle(nn.Module):
    """
    MixStyle : Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Constructor

        Args:
            p (float, optional): Probability of using MixStyle. Defaults to 0.5.
            alpha (float, optional): Parameter of the Beta distribution. Defaults to 0.3.
            eps ([type], optional): Scaling parameter to avoid numerical issues. Defaults to 1e-6.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Hidden features.

        Returns:
            torch tensor [BS x C x H x W]: Mixed features.
        """
        if not self.training:
            return x

        if np.random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class EfficientNetMixStyleEncoder(EfficientNetEncoder):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        super().__init__(stage_idxs, out_channels, model_name, depth=5)

        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.
                    x = module(x, drop_connect)

            if 0 < i < self._depth:
                x = self.mixstyle(x)

            features.append(x)

        return features
