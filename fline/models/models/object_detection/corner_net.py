import torch
import segmentation_models_pytorch as smp

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.encoders.timm import TimmEncoder
from fline.models.models.research.extractor import VectorsFromMask, VectorsFromMaskV2
from fline.models.models.research.connect_net import ConnectNet
from fline.models.models.research.connected_components import ConnectedComponents


class CornerNet(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            device,
            features_size=64,
    ):
        super(CornerNet, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
           # decoder_attention_type='scse',
        )
        self.points_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.limit = 1000
        self.threshold = 0.5
        self.connected = ConnectedComponents()
        self.extractor = VectorsFromMask(skip_zero=False)
        self.connections = ConnectNet(
            features_size=features_size,
            connections_classes=1,
            activation=self.sigmoid,
        )

    def forward(self, x, left_top_points=None, right_bottom_points=None):
        encoded = self.model(x)
        points = self.points_classes(encoded)
        points = self.sigmoid(points)
        #print(x.shape, encoded.shape,  points.shape)
        if left_top_points is None or right_bottom_points is None:
            mask = points
            left_top_points = mask[:, 0:1, :, :] > self.threshold
            right_bottom_points = mask[:, 1:2, :, :] > self.threshold
            left_top_points = self.connected(left_top_points)
            right_bottom_points = self.connected(right_bottom_points)
        left_top_vectors = self.extractor(encoded, left_top_points)
        right_bottom_vectors = self.extractor(encoded, right_bottom_points)
        connections = self.connections(left_top_vectors, right_bottom_vectors)
        return connections, points


class CornerNetTimm(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            device,
            #features_size=64,
    ):
        super(CornerNetTimm, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            #activation=None,
            #classes=features_size,
        )
        features_size = self.model.out_feature_channels
        self.points_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.limit = 1000
        self.threshold = 0.5
        self.connected = ConnectedComponents()
        self.extractor = VectorsFromMask(skip_zero=False)
        self.connections = ConnectNet(
            features_size=features_size,
            connections_classes=1,
            activation=self.sigmoid,
        )

    def forward(self, x, left_top_points=None, right_bottom_points=None):
        encoded = self.model(x)
        points = self.points_classes(encoded)
        points = self.sigmoid(points)
        #print(x.shape, encoded.shape,  points.shape)
        if left_top_points is None or right_bottom_points is None:
            mask = points
            left_top_points = mask[:, 0:1, :, :] > self.threshold
            right_bottom_points = mask[:, 1:2, :, :] > self.threshold
            left_top_points = self.connected(left_top_points)
            right_bottom_points = self.connected(right_bottom_points)
        left_top_vectors = self.extractor(encoded, left_top_points)
        right_bottom_vectors = self.extractor(encoded, right_bottom_points)
        connections = self.connections(left_top_vectors, right_bottom_vectors)
        return connections, points
