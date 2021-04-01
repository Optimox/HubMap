import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from fastai.vision.all import ConvLayer, SelfAttention, PixelShuffle_ICNR


# Multihead Vortex Atrous Attention block
class MultiheadVAA(nn.Module):
    def __init__(self, in_channels, out_channels=None, rate=1, num_heads=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.unfold = torch.nn.Unfold(3, dilation=rate, padding=rate)
        self.PE = nn.Sequential(
            nn.Conv2d(
                9, 9, 2 * (rate // 2) + 1, padding=rate // 2, groups=9, bias=False
            ),
            nn.Conv2d(9, 9, 1, bias=False),
        )
        self.out_proj = nn.Conv2d(out_channels, out_channels, 1, bias=True)

    def forward(self, q, k, v):
        shape = q.shape
        q = self.q_proj(q).view(shape[0] * self.num_heads, -1, shape[2], shape[3])
        k = self.q_proj(k).view(shape[0] * self.num_heads, -1, shape[2], shape[3])
        v = self.q_proj(v).view(shape[0] * self.num_heads, -1, shape[2], shape[3])
        shape = q.shape
        k = self.unfold(k).view(shape[0], shape[1], -1, shape[2], shape[3])
        f = (q.unsqueeze(2) * k).sum(1)
        s = torch.softmax(self.PE(f), 1)
        v = self.unfold(v).view(shape[0], shape[1], -1, shape[2], shape[3])
        o = (v * s.unsqueeze(1)).sum(2)
        shape = o.shape
        o = o.view(-1, shape[1] * self.num_heads, shape[2], shape[3])
        return self.out_proj(o)


def gem(x, p, eps=1e-6):
    t = x.type()
    x = x.float()
    x = F.avg_pool2d(
        x.clamp(min=eps).pow(p.view(1, -1, 1, 1)), (x.size(-2), x.size(-1))
    ).pow(1.0 / p.view(1, -1, 1, 1))
    x = x.type(t)
    return x


class GeM(nn.Module):
    def __init__(self, nc=1, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(nc) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.attn = nn.Sequential(
            GeM(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, groups=groups),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, groups=groups),
        )
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups)

    def forward(self, x):
        attn = torch.sigmoid(self.attn(x))
        return attn * x


class PAA(nn.Module):
    def __init__(self, in_channels, out_channels=None, rates=[1, 3, 5], num_heads=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.vaas = nn.ModuleList(
            [MultiheadVAA(in_channels, out_channels, r, num_heads) for r in rates]
        )
        self.global_attn = GlobalAttention(in_channels, out_channels)

    def forward(self, x):
        result = self.global_attn(x) + x
        x = self.bn(x)
        for vaa in self.vaas:
            result = result + vaa(x, x, x)
        return result


class CRL(nn.Module):
    def __init__(self, n_in, n_up):
        super().__init__()
        self.pool_in, self.pool_up = GeM(n_in), GeM(n_up)
        self.net = nn.Sequential(
            nn.Conv2d(n_in + n_up, n_in + n_up, 1),
            nn.BatchNorm2d(n_in + n_up),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_in + n_up, n_in, 1),
        )

    def forward(self, x, up):
        x = torch.cat([self.pool_in(x), self.pool_up(up)], 1)
        return torch.sigmoid(self.net(x))


class SRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 1, 7, padding=3))

    def forward(self, x, up):
        x = torch.cat(
            [
                x.mean(1, keepdim=True),
                up.mean(1, keepdim=True),
                x.max(1, keepdim=True)[0],
                up.max(1, keepdim=True)[0],
            ],
            1,
        )
        return torch.sigmoid(self.net(x))


class CAR(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.crl = CRL(n_channels, n_channels)
        self.srl = SRL()

    def forward(self, x, up):
        shape = x.shape
        up = F.interpolate(up, (shape[-2], shape[-1]), mode="bilinear")
        x = x * self.crl(x, up)
        up = up * self.srl(x, up)
        return x + up


class ViTBlock(nn.Module):
    def __init__(self, n=512, nheads=16, ps=0.1, pos_enc=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(n)
        self.attn = nn.MultiheadAttention(n, nheads, dropout=ps)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(n), nn.ReLU(inplace=True), nn.Conv2d(n, n, 1)
        )
        self.pos_enc = (
            None
            if pos_enc is None
            else nn.Parameter(0.1 * torch.randn(1, n, pos_enc[0], pos_enc[1]))
        )

    def forward(self, x0):
        x = self.bn(x0)
        shape = x.shape
        if self.pos_enc is not None:
            x = x + self.pos_enc
        x = x.view(shape[0], shape[1], -1).permute(2, 0, 1)
        x, w = self.attn(x, x, x)
        x = x.permute(1, 2, 0).reshape(shape)
        x0 = x0 + x
        return x0 + self.conv(x0)


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_ch * 2),
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                )
                for in_ch, out_ch in zip(input_channels, output_channels)
            ]
        )

    def forward(self, xs: list, last_layer):
        hcs = [
            F.interpolate(
                c(x), scale_factor=2 ** (len(self.convs) - i), mode="bilinear"
            )
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


def adaptive_concat_pool2d(x, sz=1):
    out1 = F.adaptive_avg_pool2d(x, sz)
    out2 = F.adaptive_max_pool2d(x, sz)
    return torch.cat([out1, out2], 1)


class UnetBlock(nn.Module):
    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        nf: int = None,
        blur: bool = False,
        self_attention: bool = False,
        **kwargs
    ):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(
            nf,
            nf,
            norm_type=None,
            xtra=SelfAttention(nf) if self_attention else None,
            **kwargs
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
