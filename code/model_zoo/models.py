import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders

from model_zoo.custom_unet import BotUnetDecoder, DoubleUNet


DECODERS = [
    "Unet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
    "UnetPlusPlus",
]
ENCODERS = list(encoders.keys())


def define_model(
    decoder_name,
    encoder_name,
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
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
    )

    if use_bot or use_fpn:
        pos_size = int(8 * (input_size / 256))

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

    if double_model:
        model = define_double_model(
            decoder_name,
            encoder_name,
            num_classes=num_classes,
            encoder_weights=encoder_weights,
        )

    return model


def define_double_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    encoder_weights="imagenet",
):
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
        in_channels=4,
    )
    double_model = DoubleUNet(model_1, model_2)
    return double_model
