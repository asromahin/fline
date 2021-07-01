import torch
import segmentation_models_pytorch as smp

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.encoders.timm import TimmEncoder
from fline.models.models.research.extractor import VectorsFromMask, VectorsFromBbox
from fline.models.models.research.connect_net import ConnectNet, ConnectNetV2, ConnectNetV3, ConnectNetV4
from fline.models.blocks.attention import SelfAttentionPatch, ConnCompAttention
from fline.models.blocks.convs import ResidualBlock


class ConnectionsSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
    ):
        super(ConnectionsSMP, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.extractor = VectorsFromMask()
        self.connector = ConnectNet(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
        )

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = self.extractor(encoded, masks)
        connections = self.connector(vectors, vectors)
        return vectors, connections


class TrackConnectionsSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(TrackConnectionsSMP, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.Unet(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox(
            merge_type='max',
        )
        self.connector = ConnectNet(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x1, x2, box1, box2):
        encoded1 = self.model(x1)
        encoded2 = self.model(x2)
        #encoded1 = self.relu(encoded1)
        #encoded2 = self.relu(encoded2)
        vectors1 = self.extractor(encoded1, box1)
        vectors2 = self.extractor(encoded2, box2)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections


class TrackConnectionsTimm(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            #features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
            features_block=None,
    ):
        super(TrackConnectionsTimm, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            features_block=features_block,
            #activation=None,
            #classes=features_size,
        )
        features_size = self.model.out_feature_channels
        self.attn = ConnCompAttention(features_size)
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox(
            merge_type='max',
        )
        self.connector = ConnectNetV4(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        encoded1 = self.model(x1)
        encoded2 = self.model(x2)
        encoded1, attn_encoded1 = self.attn(encoded1)
        encoded2, attn_encoded2 = self.attn(encoded2)
        vectors1 = self.extractor(encoded1, box1)
        vectors2 = self.extractor(encoded2, box2)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections


class TrackConnectionsSMPV2(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(TrackConnectionsSMPV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox()
        self.connector = ConnectNet(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        im = torch.cat([x1, x2], dim=2)
        encoded = self.model(im)
        box2[:,:,1] += x1.shape[2]
        vectors1 = self.extractor(encoded, box1)
        vectors2 = self.extractor(encoded, box2)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections


class TrackConnectionsSMPV3(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(TrackConnectionsSMPV3, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.atn = SelfAttentionPatch(in_dim=features_size, downsample=16)
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox(
            merge_type='matmul',
        )
        self.conv = ResidualBlock(in_channels=features_size, out_channels=features_size)
        self.linear = torch.nn.Linear(features_size**2, features_size)
        self.connector = ConnectNet(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        encoded1 = self.model(x1)
        encoded2 = self.model(x2)
        encoded1, attn1 = self.atn(encoded1)
        encoded2, attn2 = self.atn(encoded2)
        encoded1 = self.conv(encoded1)
        encoded2 = self.conv(encoded2)
        vectors1 = self.extractor(encoded1, box1)
        vectors2 = self.extractor(encoded2, box2)
        #print(vectors1.shape)
        vectors1 = self.linear(vectors1.permute(0,2,1,3).mean(dim=3)).permute(0,2,1).unsqueeze(dim=3)
        vectors2 = self.linear(vectors2.permute(0,2,1,3).mean(dim=3)).permute(0,2,1).unsqueeze(dim=3)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections


class TrackConnectionsTimmV2(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            #features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
            features_block=None,
    ):
        super(TrackConnectionsTimmV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            features_block=features_block,
            #activation=None,
            #classes=features_size,
        )
        features_size = self.model.out_feature_channels
        self.atn = SelfAttentionPatch(in_dim=features_size, downsample=16)
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox(
            merge_type='max',
        )
        self.connector = ConnectNetV3(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        encoded1 = self.model(x1)
        encoded2 = self.model(x2)
        encoded1, attn1 = self.atn(encoded1)
        encoded2, attn2 = self.atn(encoded2)
        vectors1 = self.extractor(encoded1, box1)
        vectors2 = self.extractor(encoded2, box2)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections


class TrackEncoderModel(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            #features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(TrackEncoderModel, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmEncoder(self.backbone_name)
        features_size = self.model.features_channels[-1]
        self.atn = SelfAttentionPatch(in_dim=features_size, downsample=2)
        self.connections_classes = connections_classes
        self.connector = ConnectNet(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        bvectors1 = []
        bvectors2 = []
        for b in range(x1.shape[0]):
            vectors1 = []
            for i in range(box1.shape[1]):
                box = box1[b, i].to(dtype=torch.long)
                #print(box.shape)
                encoded1 = self.model(x1[b:b+1, :, box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
                #encoded1, attn1 = self.atn(encoded1)
                encoded1 = encoded1.mean(dim=0)
                encoded1 = encoded1.mean(dim=1).mean(dim=1)
                vectors1.append(encoded1.unsqueeze(dim=0))
            vectors1 = torch.cat(vectors1, dim=0)
            vectors2 = []
            for i in range(box2.shape[1]):
                box = box2[b, i].to(dtype=torch.long)
                encoded2 = self.model(x2[b:b+1, :, box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
                #encoded2, attn2 = self.atn(encoded2)
                encoded2 = encoded2.mean(dim=0)
                encoded2 = encoded2.mean(dim=1).mean(dim=1)
                vectors2.append(encoded2.unsqueeze(dim=0))
            vectors2 = torch.cat(vectors2, dim=0)
            bvectors1.append(vectors1.unsqueeze(0))
            bvectors2.append(vectors2.unsqueeze(0))
        bvectors1 = torch.cat(bvectors1, dim=0).unsqueeze(3).transpose(1,2)
        bvectors2 = torch.cat(bvectors2, dim=0).unsqueeze(3).transpose(1,2)
        connections = self.connector(bvectors1, bvectors2)
        return bvectors1, bvectors2, connections


class TrackConnectionsSMPV4(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(TrackConnectionsSMPV4, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.Unet(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.relu = torch.nn.ReLU()
        self.connections_classes = connections_classes
        self.extractor = VectorsFromBbox(
            merge_type='max',
        )
        self.connector = ConnectNetV2(
            features_size=features_size,
            connections_classes=connections_classes,
            activation=activation,
            n_layers=n_layers,
        )

    def forward(self, x1, x2, box1, box2):
        encoded1 = self.model(x1)
        encoded1 = self.relu(encoded1)
        encoded2 = self.model(x2)
        encoded2 = self.relu(encoded2)
        vectors1 = self.extractor(encoded1, box1)
        vectors2 = self.extractor(encoded2, box2)
        connections = self.connector(vectors1, vectors2)
        return vectors1, vectors2, connections
