import torch
from fline.models.encoders.timm import TimmEncoder
from fline.models.decoders.dynamic import DynamicDecoder
from fline.models.blocks.convs import ResidualBlock


class TextStyleBrush(torch.nn.Module):
    def __init__(
            self,
            gen_encoder_backbone_name='resnet18',
            original_encoder_backbone_name='resnet18',
    ):
        super(TextStyleBrush, self).__init__()
        self.gen_encoder = TimmEncoder(
            backbone_name=gen_encoder_backbone_name,
            output_features=True,
        )
        self.original_encoder = TimmEncoder(
            backbone_name=original_encoder_backbone_name,
            output_features=True,
        )
        assert len(self.original_encoder.output_features) == self.gen_encoder.output_features
        features = [
            self.original_encoder.output_features[i] + self.gen_encoder.output_features[i]
            for i in range(len(self.original_encoder.output_features))
        ]
        self.decoder = DynamicDecoder(
            features=features,
            connection_type='cat',
        )
        self.output_decoder = torch.nn.Sequential(
            ResidualBlock(
                in_channels=features[0],
                out_channels=features[0],
                stride=1,
            ),
            torch.nn.Conv2d(
            in_channels=features[0],
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, gen_x, gen_y):
        x_features = self.original_encoder(x)
        gen_x_features = self.gen_encoder(gen_x)
        gen_y_features = self.gen_encoder(gen_y)
        gen_x_result = self.decoder(self.merge_features(x_features, gen_x_features))
        gen_y_result = self.decoder(self.merge_features(x_features, gen_y_features))
        return gen_x_result, gen_y_result

    def merge_features(self, f1, f2):
        return [torch.cat([f1[i], f2[i]], dim=1) for i in range(len(f1))]

