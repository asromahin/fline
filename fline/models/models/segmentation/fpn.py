import torch


from fline.models.encoders.timm import TimmEncoder
from fline.models.decoders.dynamic import DynamicDecoder
from fline.models.blocks.convs import SimpleConvBlock, SimpleUpConvBlock


class FPN(torch.nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            feature_channels,
            features_block=None,
            classes=None,
            activation=None,
            return_dict=True,
    ):
        super(FPN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.features_block = features_block
        self.classes = classes
        self.out_feature_channels = sum([*feature_channels[:-1], feature_channels[0]])
        self.classify = torch.nn.Conv2d(
            kernel_size=3,
            stride=1,
            padding=1,
            in_channels=self.out_feature_channels,
            out_channels=self.classes,
        )
        #print(feature_channels)
        self.activation = activation
        self.return_dict = return_dict
        self.upsample_feature = torch.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        #print(x.shape)
        features = self.encoder(x)
        if self.features_block is not None:
            features = self.features_block(features)
        decoder_features = self.decoder(features)
        x = None
        for decoder_feature in decoder_features:
            if x is None:
                x = decoder_feature
            else:
                x = torch.cat([self.upsample_feature(x), decoder_feature], dim=1)
                #print(decoder_feature.shape, x.shape)

        #x = decoder_features[-1]
        #print(x.shape)
        out = {
            'out': x,
            'encoder_features': features,
            'decoder_features': decoder_features,
        }
        #print(self.classes, self.activation)
        if self.classes is not None:
            cls_x = self.classify(x)
            if self.activation is not None:
                cls_x = self.activation(cls_x)
                out['classes'] = cls_x
        if self.return_dict:
            return out
        else:
            if 'classes' in out.keys():
                return out['classes']
            else:
                return out['out']


class TimmFPN(FPN):
    def __init__(
            self,
            backbone_name,
            features_block=None,
            classes=None,
            activation=None,
            return_dict=False,
    ):
        encoder = TimmEncoder(backbone_name=backbone_name, output_features=True)
        if features_block == 'each':
            features_block = EachForEachFeaturesBlock(
                upsample=torch.nn.UpsamplingNearest2d(scale_factor=2),
                pool=torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                features_channels=encoder.features_channels,
                conv=SimpleConvBlock,
            )
        else:
            features_block = None
        decoder = DynamicDecoder(
            features=encoder.features_channels,
            connection_type='add',
            return_features=True,
        )
        self.out_channels = encoder.features_channels[0]
        super(TimmFPN, self).__init__(
            encoder=encoder,
            decoder=decoder,
            features_block=features_block,
            classes=classes,
            activation=activation,
            feature_channels=encoder.features_channels,
            return_dict=return_dict,
        )


class EachForEachFeaturesBlock(torch.nn.Module):
    def __init__(
            self,
            upsample,
            pool,
            features_channels,
            conv=SimpleConvBlock,
    ):
        super(EachForEachFeaturesBlock, self).__init__()
        self.upsample = upsample
        self.pool = pool
        self.features_channels = features_channels
        for i in range(len(self.features_channels)):
            for j in range(i+1, len(self.features_channels)):
                key1 = 'conv_'+str(i) + '_' + str(j)
                cur_conv1 = conv(
                    in_channels=self.features_channels[i],
                    out_channels=self.features_channels[j],
                )
                setattr(self, key1, cur_conv1)
                key2 = 'conv_' + str(j) + '_' + str(i)
                cur_conv2 = conv(
                    in_channels=self.features_channels[j],
                    out_channels=self.features_channels[i],
                )
                setattr(self, key2, cur_conv2)

    def forward(self, features):
        #print('-'*60)
        for i in range(len(self.features_channels)):
            for j in range(i+1, len(self.features_channels)):
                key1 = 'conv_' + str(i) + '_' + str(j)
                cur_conv1 = getattr(self, key1)
                key2 = 'conv_' + str(j) + '_' + str(i)
                cur_conv2 = getattr(self, key2)
                #print(i, j)
                cur_feature = cur_conv2(features[j])
                for t in range(j - i):
                    cur_feature = self.upsample(cur_feature)
                features[i] = features[i] + cur_feature

                cur_feature = cur_conv1(features[i])
                for t in range(j - i):
                    cur_feature = self.pool(cur_feature)
                features[j] = features[j] + cur_feature

        return features

