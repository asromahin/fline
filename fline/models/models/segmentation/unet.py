import torch


from fline.models.encoders.timm import TimmEncoder
from fline.models.decoders.dynamic import DynamicDecoder


class Unet(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            return_encoder_features=False,
    ):
        super(Unet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.return_encoder_features = return_encoder_features

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        if self.return_encoder_features:
            return x, features
        else:
            return x


class TimmUnet(Unet):
    def __init__(
            self,
            backbone_name,
            return_features=False,
    ):
        encoder = TimmEncoder(backbone_name=backbone_name, output_features=True)
        decoder = DynamicDecoder(
            features=encoder.features_channels,
            connection_type='cat',
            return_features=return_features,
        )
        super(TimmUnet, self).__init__(
            encoder=encoder,
            decoder=decoder,
            return_encoder_features=False,
        )
