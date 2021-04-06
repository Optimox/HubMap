import torch.nn as nn

from model_zoo.modules import FPN
from model_zoo.bot import BotBlock
from segmentation_models_pytorch.unet.decoder import DecoderBlock


class BotUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            use_bot=False,
            pos_enc=(256, 256),
            use_fpn=False,
    ):
        super().__init__()

        self.use_fpn = use_fpn
        self.use_bot = use_bot

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        nc = encoder_channels[0]
        nt = nc

        self.center = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.Conv2d(nc, nt, 1, bias=False),
            BotBlock(nt, pos_enc, nheads=4),
            nn.Conv2d(nt, nt, 1, bias=False),
            nn.BatchNorm2d(nt),
            nn.ReLU(inplace=True),
            nn.Conv2d(nt, nt, 1),
        )

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        if use_fpn:
            self.blocks = nn.ModuleList(blocks[:-1])  # remove last for fpn

            fpn_fts = [nt] + list(decoder_channels[:3])
            self.fpn = FPN(fpn_fts, [decoder_channels[-1]] * 4)

            self.final_block = DecoderBlock(
                decoder_channels[-1] * 4 + decoder_channels[3],
                0,
                decoder_channels[-1],
                **kwargs
            )

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        if self.use_bot:
            x = self.center(x)

        dec = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            dec.append(x)

        if self.use_fpn:
            x = self.fpn(dec[:-1], dec[-1])
            x = self.final_block(x)

        return x
