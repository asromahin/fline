import torch
import timm

from fline.models.encoders.base import BaseEncoder


class TimmEncoder(BaseEncoder):
    def __init__(self, backbone_name, output_features=False):
        super(TimmEncoder, self).__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True)
        self.features_channels = self._get_features()
        self.output_features = output_features

    def _get_features(self):
        with torch.no_grad():
            features = self.encoder(torch.zeros((1, 3, 64, 64)))
        return [f.shape[1] for f in features]

    def forward(self, x):
        features = self.encoder(x)
        if self.output_features:
            return features
        else:
            return features[-1]
