import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders
import torch


DECODERS = ["Unet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN"]
ENCODERS = list(encoders.keys())


def define_model(
    decoder_name, encoder_name, num_classes=1, activation=None, encoder_weights="imagenet", double_model=False
):
    """
    Loads a segmentation architecture

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
    
    if double_model:
        model = define_double_model(decoder_name,
                                    encoder_name,
                                    num_classes=num_classes,
                                    activation=activation, encoder_weights=encoder_weights)
    else:
        model = decoder(
            encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )

    return model


class DoubleUNet(torch.nn.Module):
    def __init__(self, model_1, model_2):
        super(DoubleUNet, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        
    def forward(self, x):
        out_1 = torch.sigmoid(self.model_1(x))
        x_2 = torch.cat([out_1, x], dim=1)
        out_2 = self.model_2(x_2)
        return out_2
    

def define_double_model(
    decoder_name, encoder_name, num_classes=1, activation=None, encoder_weights="imagenet"
):
    """
    Loads a segmentation architecture

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

    model_1 = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
    )
    
    model_2 = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        in_channels=4
        
    )

    double_model = DoubleUNet(model_1, model_2)
    return double_model