import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import ConvLayer

from model_zoo.modules import PAA, UnetBlock, FPN, ViTBlock, CAR
from model_zoo.bot import BotBlock


class UneXt50_PCA(nn.Module):
    def __init__(self, ps=0.0, **kwargs):
        super().__init__()
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        nc, nt = 2048, 256
        self.paa = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.Conv2d(nc, nt, 1, bias=False),
            PAA(nt, num_heads=16),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, 256, 1),
        )
        # self.paa = nn.Sequential(nn.BatchNorm2d(nc),nn.Conv2d(nc,nt,1,bias=False),
        #      ViTBlock(nt,ps=0.1,nheads=16),nn.Conv2d(nt,nt,1,bias=False),
        #      nn.BatchNorm2d(nt),nn.ReLU(inplace=True),nn.Conv2d(nt,256,1))
        self.car = CAR(256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(ps / 4),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        up = self.paa(enc4)
        x = self.final_conv(self.car(enc1, up))
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        return x

    split_layers = lambda m: [  # noqa
        list(m.module.enc0.parameters())
        + list(m.module.enc1.parameters())
        + list(m.module.enc2.parameters())
        + list(m.module.enc3.parameters())
        + list(m.module.enc4.parameters()),
        list(m.module.paa.parameters())
        + list(m.module.car.parameters())
        + list(m.module.final_conv.parameters()),
    ]


class UneXt50_VAA(nn.Module):
    def __init__(self, nt=512, pos_enc=None, **kwargs):
        super().__init__()
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        nc = 2048
        self.paa = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.Conv2d(nc, nt, 1, bias=False),
            PAA(nt, num_heads=16),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )

        self.dec4 = UnetBlock(nt, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.15)
        self.final_conv = ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.paa(enc4)
        dec3 = self.dec4(enc5, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

    split_layers = lambda m: [  # noqa
        list(m.module.enc0.parameters())
        + list(m.module.enc1.parameters())
        + list(m.module.enc2.parameters())
        + list(m.module.enc3.parameters())
        + list(m.module.enc4.parameters()),
        list(m.module.paa.parameters())
        + list(m.module.dec4.parameters())
        + list(m.module.dec3.parameters())
        + list(m.module.dec2.parameters())
        + list(m.module.dec1.parameters())
        + list(m.module.fpn.parameters())
        + list(m.module.final_conv.parameters()),
    ]


class UneXt50_ViT(nn.Module):
    def __init__(self, nt=512, pos_enc=None, **kwargs):
        super().__init__()
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        nc = 2048
        self.vit = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.Conv2d(nc, nt, 1, bias=False),
            ViTBlock(nt, ps=0.1, nheads=16, pos_enc=pos_enc),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )

        self.dec4 = UnetBlock(nt, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.15)
        self.final_conv = ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.vit(enc4)
        dec3 = self.dec4(enc5, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

    split_layers = lambda m: [  # noqa
        list(m.module.enc0.parameters())
        + list(m.module.enc1.parameters())
        + list(m.module.enc2.parameters())
        + list(m.module.enc3.parameters())
        + list(m.module.enc4.parameters()),
        list(m.module.vit.parameters())
        + list(m.module.dec4.parameters())
        + list(m.module.dec3.parameters())
        + list(m.module.dec2.parameters())
        + list(m.module.dec1.parameters())
        + list(m.module.fpn.parameters())
        + list(m.module.final_conv.parameters()),
    ]


class UneXt50_BoT(nn.Module):
    def __init__(self, nt=512, pos_enc=(16, 16), **kwargs):
        super().__init__()
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        nc1, nc2 = 2048, 1024
        self.vit1 = nn.Sequential(
            nn.BatchNorm2d(nc1),
            nn.Dropout2d(0.1),
            nn.Conv2d(nc1, nt, 1, bias=False),
            BotBlock(nt, pos_enc, nheads=4),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )
        self.vit2 = nn.Sequential(
            nn.BatchNorm2d(nc2),
            nn.Dropout2d(0.1),
            nn.Conv2d(nc2, nt, 1, bias=False),
            BotBlock(nt, (pos_enc[0] * 2, pos_enc[1] * 2), nheads=4),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )

        self.dec4 = UnetBlock(nt, nt, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([nt, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.15)
        self.final_conv = ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.vit1(self.enc4(enc3))
        enc3 = self.vit2(enc3)
        dec3 = self.dec4(enc4, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc4, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

    split_layers = lambda m: [  # noqa
        list(m.module.enc0.parameters())
        + list(m.module.enc1.parameters())
        + list(m.module.enc2.parameters())
        + list(m.module.enc3.parameters())
        + list(m.module.enc4.parameters()),
        list(m.module.vit1.parameters())
        + list(m.module.vit2.parameters())
        + list(m.module.dec4.parameters())
        + list(m.module.dec3.parameters())
        + list(m.module.dec2.parameters())
        + list(m.module.dec1.parameters())
        + list(m.module.fpn.parameters())
        + list(m.module.final_conv.parameters()),
    ]


class UneXt50_BoT1(nn.Module):
    def __init__(self, nt=512, pos_enc=(16, 16), **kwargs):
        super().__init__()
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        nc = 2048
        self.vit = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.Conv2d(nc, nt, 1, bias=False),
            BotBlock(nt, pos_enc, nheads=4),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )

        self.dec4 = UnetBlock(nt, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([nt, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.15)
        self.final_conv = ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.vit(self.enc4(enc3))
        dec3 = self.dec4(enc4, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc4, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x

    split_layers = lambda m: [  # noqa
        list(m.module.enc0.parameters())
        + list(m.module.enc1.parameters())
        + list(m.module.enc2.parameters())
        + list(m.module.enc3.parameters())
        + list(m.module.enc4.parameters()),
        list(m.module.vit.parameters())
        + list(m.module.dec4.parameters())
        + list(m.module.dec3.parameters())
        + list(m.module.dec2.parameters())
        + list(m.module.dec1.parameters())
        + list(m.module.fpn.parameters())
        + list(m.module.final_conv.parameters()),
    ]
