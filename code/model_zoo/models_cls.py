import torch.nn as nn
import segmentation_models_pytorch

from efficientnet_pytorch.utils import Conv2dStaticSamePadding

from model_zoo.models import ENCODERS, DECODERS
from utils.torch import load_model_weights


class SegmentationModelCls(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = nn.AdaptiveAvgPool2d(1)

        self.nb_ft = model.encoder.out_channels[-1]
        self.logits = nn.Linear(self.nb_ft, num_classes)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pooler(x).view(-1, self.nb_ft)
        return self.logits(x)


def define_model_cls(
    decoder_name,
    encoder_name,
    in_channels=3,
    num_classes=1,
    activation=None,
    cp_path="",
):
    """
    Loads a segmentation architecture.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
    Returns:
        torch model -- Pretrained model.
    """

    assert decoder_name in DECODERS, "Decoder name not supported"
    assert encoder_name in ENCODERS, "Encoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)

    model = decoder(
        encoder_name,
        encoder_weights=None,
        classes=num_classes,
        activation=activation,
    )

    if cp_path:
        model = load_model_weights(model, cp_path)

    model = SegmentationModelCls(model, num_classes=num_classes)

    if in_channels != 3:
        if "efficientnet" in encoder_name:
            conv = model.encoder._conv_stem
            new_conv = Conv2dStaticSamePadding(
                in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                dilation=conv.dilation,
                groups=conv.groups,
                image_size=128,
            )
            new_conv.static_padding = conv.static_padding

            model.encoder._conv_stem = new_conv

        else:  # TODO : support resnets
            raise NotImplementedError

    return model
