import torch

from fline.models.encoders.timm import TimmEncoder
from fline.models.decoders.dynamic import DynamicDecoder


class ConvAE(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            return_botleneck=False,
    ):
        super(ConvAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.return_botleneck = return_botleneck

    def forward(self, x):
        feature = self.encoder(x)
        x = self.decoder(feature)
        if self.return_botleneck:
            return x, feature
        else:
            return x


class TimmConvAE(ConvAE):
    def __init__(
            self,
            backbone_name,
            return_botleneck=False,
    ):
        encoder = TimmEncoder(backbone_name=backbone_name, output_features=False)
        decoder = DynamicDecoder(features=encoder.features_channels)
        super(TimmConvAE, self).__init__(
            encoder=encoder,
            decoder=decoder,
            return_botleneck=return_botleneck,
        )
