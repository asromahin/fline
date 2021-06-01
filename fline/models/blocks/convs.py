import torch


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SimpleUpConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            kernel_size,
            padding,
    ):
        super(SimpleUpConvBlock, self).__init__()

        self.conv = torch.nn.ConvTranspose2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
            output_padding=1,
        )
        self.conv2 = ResidualBlock(out_features, out_features)
        # self.uppool = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        # x = self.uppool(x)
        return x