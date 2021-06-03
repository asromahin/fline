import torch

from fline.models.blocks.convs import SimpleUpConvBlock


class DynamicDecoder(torch.nn.Module):
    def __init__(
            self,
            conv_block=SimpleUpConvBlock,
            features=[3, 8, 16, 32, 64],
            return_features=False,
            connection_type=None,
    ):
        super(DynamicDecoder, self).__init__()
        self.return_features = return_features
        self.connection_type = connection_type
        kernel_size = (3, 3)
        padding = (1, 1)
        self.convs = []
        for feature_num, feature in enumerate(features[:-1]):
            in_features = features[len(features) - 1 - feature_num]
            if self.connection_type == 'cat':
                in_features *= 2
            out_features = features[len(features) - 2 - feature_num]
            conv = conv_block(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                padding=padding,
            )
            key = 'upconv_' + str(feature_num)
            self.convs.append(key)
            setattr(self, key, conv)
        self.last_conv = conv_block(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=kernel_size,
                padding=padding,
            )

    def forward(self, features):
        if self.return_features:
            out = []
        x = features[-1]
        for i, conv_key in enumerate(self.convs):
            conv = getattr(self, conv_key)
            if i > 0 and self.connection_type is not None:
                if self.connection_type == 'add':
                    x = x + features[len(features)-1-i]
                elif self.connection_type == 'cat':
                    x = torch.cat([x, features[len(features)-1-i]], dim=1)
            x = conv(x)
            if self.return_features:
                out.append(x)
        x = self.last_conv(x)
        if self.return_features:
            out.append(x)
            return out
        return x
