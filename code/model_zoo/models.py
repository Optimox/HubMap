import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders
import torch
import model_zoo.unext
from model_zoo.bot_unet import BotUnetDecoder


DECODERS = ["Unet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN", "BoTUnet"]
ENCODERS = list(encoders.keys())


def define_model(
    decoder_name,
    encoder_name,
    model_name,
    num_classes=1,
    activation=None,
    encoder_weights="imagenet",
    input_size=256,
    double_model=False,
    use_bot=False,
    use_fpn=False,
):
    """
    Loads a segmentation architecture.
    model_name overides decoder_name and encoder_name when defined.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        model_name (str): Model name if the model is not defined using SMP.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
    Returns:
        torch model -- Pretrained model.
    """
    pos_size = int(8 * (input_size // 256))

    if not model_name:
        assert decoder_name in DECODERS, "Decoder name not supported"
        assert encoder_name in ENCODERS, "Encoder name not supported"

        decoder = getattr(segmentation_models_pytorch, decoder_name)
        if double_model:
            model = define_double_model(decoder_name,
                                        encoder_name,
                                        model_name,
                                        num_classes=num_classes,
                                        activation=activation,
                                        encoder_weights=encoder_weights)
        else:
            model = decoder(
                encoder_name,
                encoder_weights=encoder_weights,
                classes=num_classes,
                activation=activation,
            )
    else:
        pos_size = int(8 * (input_size // 256))
        model = getattr(model_zoo.unext, model_name)(pos_enc=(pos_size, pos_size))

        model = decoder(
            encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
        )

        if use_bot or use_fpn:
            model.decoder = BotUnetDecoder(
                encoder_channels=model.encoder.out_channels,
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_batchnorm=True,
                center=False,
                attention_type=None,
                use_bot=use_bot,
                pos_enc=(pos_size, pos_size),
                use_fpn=use_fpn,
            )
        else:
            model = getattr(model_zoo.unext, model_name)()

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

def define_double_model(decoder_name,
                        encoder_name,
                        model_name,
                        num_classes=1,
                        activation=None,
                        encoder_weights="imagenet"):
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
        in_channels=4)
    double_model = DoubleUNet(model_1, model_2)
    return double_model