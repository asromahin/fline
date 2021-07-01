import torch
import segmentation_models_pytorch as smp

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.encoders.timm import TimmEncoder
from fline.models.models.research.extractor import VectorsFromMask
from fline.models.models.research.connect_net import ConnectNet


class ObjectDetectionModel(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            max_objects=32,
            features_block=None,
    ):
        super(ObjectDetectionModel, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            classes=max_objects,
            activation=torch.nn.Softmax2d(),
            features_block=features_block,
            return_dict=True,
        )
        self.bbox_layer = torch.nn.Linear(self.model.out_feature_channels, self.model.out_feature_channels//2)
        self.bbox_layer_out = torch.nn.Linear(self.model.out_feature_channels//2, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        res_dict = self.model(x)
        encoded = res_dict['out']
        encoded_classes = res_dict['classes']
        connected_out = encoded_classes.argmax(dim=1)
        out_bboxes = []
        zero_bbox = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(connected_out):
            cur_mask = (connected_out == i).to(dtype=torch.float32)
            #print(cur_mask.shape, encoded.shape)
            vector_encoded = cur_mask * encoded
            vector_encoded = vector_encoded.max(dim=2)[0]
            vector_encoded = vector_encoded.max(dim=2)[0]
            vector_encoded = torch.transpose(vector_encoded, 0, 1)
            bbox = self.bbox_layer(vector_encoded)
            bbox = self.bbox_layer_out(bbox)
            bbox = bbox.unsqueeze(1)
            bbox = self.sigmoid(bbox)
            #print(bbox)
            if i != 0:
                out_bboxes.append(bbox)
            else:
                zero_bbox.append(bbox)
        if len(out_bboxes) == 0:
            out_bboxes = zero_bbox
        out_bboxes = torch.cat(out_bboxes, dim=1)
        return out_bboxes, encoded_classes


class ObjectDetectionModelV2(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_block=None,
    ):
        super(ObjectDetectionModelV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            features_block=features_block,
            return_dict=True,
        )
        self.bbox_layer = torch.nn.Linear(self.model.out_feature_channels, self.model.out_feature_channels//2)
        self.bbox_layer_out = torch.nn.Linear(self.model.out_feature_channels//2, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        res_dict = self.model(x)
        encoded = res_dict['out']
        #encoded_classes = res_dict['classes']
        #connected_out = encoded_classes.argmax(dim=1)
        out_bboxes = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i).to(dtype=torch.float32)[:, 0, :, :]
                vector_encoded = cur_mask * encoded
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = torch.transpose(vector_encoded, 0, 1)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = bbox.unsqueeze(1)
                bbox = self.sigmoid(bbox)
                out_bboxes.append(bbox)
        out_bboxes = torch.cat(out_bboxes, dim=1)
        return out_bboxes


class ObjectDetectionModelV3(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_block=None,
    ):
        super(ObjectDetectionModelV3, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            features_block=features_block,
            return_dict=True,
        )
        self.bbox_layer = torch.nn.Linear(self.model.out_feature_channels, self.model.out_feature_channels//2)
        self.bbox_layer_out = torch.nn.Linear(self.model.out_feature_channels//2, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, bboxes):
        res_dict = self.model(x)
        encoded = res_dict['out']
        out_bboxes = []
        encoded = torch.transpose(encoded, 0, 1)
        #print('-'*60)

        for b, batch_boxes in enumerate(bboxes):
            cur_bboxes = []
            for i, target_box in enumerate(batch_boxes):
                x,y,w,h = target_box
                x = int(x * encoded.shape[3])
                w = int(w * encoded.shape[3])
                y = int(y * encoded.shape[2])
                h = int(h * encoded.shape[2])
                #print(x,y,w,h)
                vector_encoded = encoded[:, b:b+1, y-h//2: y+h//2+1, x-w//2: x+w//2+1]
                #print(vector_encoded.shape)
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = torch.transpose(vector_encoded, 0, 1)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                #bbox = bbox.unsqueeze(0)
                bbox = self.sigmoid(bbox)
                cur_bboxes.append(bbox)
            cur_bboxes = torch.cat(cur_bboxes, dim=0)
            cur_bboxes = cur_bboxes.unsqueeze(1)
            out_bboxes.append(cur_bboxes)
        out_bboxes = torch.cat(out_bboxes, dim=1)
        return out_bboxes


class ObjectDetectionModelSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
    ):
        super(ObjectDetectionModelSMP, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.Unet(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.bbox_layer = torch.nn.Linear(features_size, features_size//2)
        self.bbox_layer_out = torch.nn.Linear(features_size//2, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        #print(encoded.shape)
        out_bboxes = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i).to(dtype=torch.float32)[:, 0, :, :]
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = cur_mask * encoded
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = torch.transpose(vector_encoded, 0, 1)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = bbox.unsqueeze(1)
                bbox = self.sigmoid(bbox)
                #print(bbox)
                out_bboxes.append(bbox)
        out_bboxes = torch.cat(out_bboxes, dim=1)
        return out_bboxes


class ConnectionsSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
    ):
        super(ConnectionsSMP, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.bbox_layer = torch.nn.Linear(features_size*2, features_size//2)
        self.bbox_layer_out = torch.nn.Linear(features_size//2, connections_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i).to(dtype=torch.float32)[:, 0, :, :]
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = cur_mask * encoded
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = vector_encoded.max(dim=2)[0]
                vector_encoded = torch.transpose(vector_encoded, 0, 1).unsqueeze(1)
                vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        connections = []
        for i in range(vectors.shape[1]):
            line_connections = []
            for j in range(vectors.shape[1]):
                vector1 = vectors[:, i, :]
                vector2 = vectors[:, j, :]
                #vector_encoded = vector1+vector2
                vector_encoded = torch.cat([vector1, vector2], dim=1)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = self.sigmoid(bbox)
                bbox = bbox.unsqueeze(2)
                line_connections.append(bbox)

            line_connections = torch.cat(line_connections, dim=2).unsqueeze(3)
            connections.append(line_connections)
        connections = torch.cat(connections, dim=3)
        #print(connections.shape)
        return vectors, connections


class ConnectionsSMPV2(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
    ):
        super(ConnectionsSMPV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.bbox_layer = torch.nn.Linear(features_size*2, features_size//2)
        self.bbox_layer_out = torch.nn.Linear(features_size//2, connections_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i)[:, 0, :, :]
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = []
                for i, cm in enumerate(cur_mask):
                    v_encoded = encoded[:, i, cm]
                    v_encoded = v_encoded.max(dim=1)[0]
                    vector_encoded.append(v_encoded.unsqueeze(0))
                vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
                vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        connections = []
        for i in range(vectors.shape[1]):
            line_connections = []
            for j in range(vectors.shape[1]):
                vector1 = vectors[:, i, :]
                vector2 = vectors[:, j, :]
                #vector_encoded = vector1+vector2
                vector_encoded = torch.cat([vector1, vector2], dim=1)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = self.sigmoid(bbox)
                bbox = bbox.unsqueeze(2)
                line_connections.append(bbox)

            line_connections = torch.cat(line_connections, dim=2).unsqueeze(3)
            connections.append(line_connections)
        connections = torch.cat(connections, dim=3)
        #print(connections.shape)
        return vectors, connections


class ConnectionsTimm(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            #features_size=64,
            connections_classes=4,
            features_block=None,
    ):
        super(ConnectionsTimm, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            activation=None,
            features_block=features_block,
            #classes=features_size,
        )
        self.connections_classes = connections_classes
        features_size = self.model.out_feature_channels
        #print(features_size)
        self.bbox_layer = torch.nn.Linear(features_size*2, features_size//2)
        self.bbox_layer_out = torch.nn.Linear(features_size//2, connections_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i)[:, 0, :, :]
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = []
                for i, cm in enumerate(cur_mask):
                    v_encoded = encoded[:, i, cm]
                    v_encoded = v_encoded.max(dim=1)[0]
                    vector_encoded.append(v_encoded.unsqueeze(0))
                vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
                vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        connections = []
        for i in range(vectors.shape[1]):
            line_connections = []
            for j in range(vectors.shape[1]):
                vector1 = vectors[:, i, :]
                vector2 = vectors[:, j, :]
                #vector_encoded = vector1+vector2
                vector_encoded = torch.cat([vector1, vector2], dim=1)
                #print(vector_encoded.shape)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = self.sigmoid(bbox)
                bbox = bbox.unsqueeze(2)
                line_connections.append(bbox)

            line_connections = torch.cat(line_connections, dim=2).unsqueeze(3)
            connections.append(line_connections)
        connections = torch.cat(connections, dim=3)
        #print(connections.shape)
        return vectors, connections


class ConnectionsFullEncode(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            connections_classes=4,
    ):
        super(ConnectionsFullEncode, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmEncoder(
            backbone_name=backbone_name,
            in_channels=4,
        )
        features_size = self.model.features_channels[-1]
        print(features_size)
        self.connections_classes = connections_classes
        self.bbox_layer = torch.nn.Linear(features_size*2, features_size//2)
        self.bbox_layer_out = torch.nn.Linear(features_size//2, connections_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        vectors = []
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i)
                new_x = torch.cat([x, cur_mask], dim=1)
                encoded = self.model(new_x)
                #print(cur_mask.shape, encoded.shape)
                encoded = encoded.max(dim=2)[0].max(dim=2)[0].unsqueeze(1)
                #print(encoded.shape)
                vectors.append(encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        #print(vectors.shape)
        connections = []
        for i in range(vectors.shape[1]):
            line_connections = []
            for j in range(vectors.shape[1]):
                vector1 = vectors[:, i, :]
                vector2 = vectors[:, j, :]
                #vector_encoded = vector1+vector2
                vector_encoded = torch.cat([vector1, vector2], dim=1)
                #print(vector_encoded.shape)
                bbox = self.bbox_layer(vector_encoded)
                bbox = self.bbox_layer_out(bbox)
                bbox = self.sigmoid(bbox)
                bbox = bbox.unsqueeze(2)
                line_connections.append(bbox)

            line_connections = torch.cat(line_connections, dim=2).unsqueeze(3)
            connections.append(line_connections)
        connections = torch.cat(connections, dim=3)
        #print(connections.shape)
        return vectors, connections


class ConnectionsSMPV3(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
    ):
        super(ConnectionsSMPV3, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.bbox_layer = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=connections_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in torch.unique(masks):
            if i != 0:
                cur_mask = (masks == i)[:, 0, :, :]
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = []
                for i, cm in enumerate(cur_mask):
                    v_encoded = encoded[:, i, cm]
                    v_encoded = v_encoded.max(dim=1)[0]
                    vector_encoded.append(v_encoded.unsqueeze(0))
                vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
                vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        vectors = vectors.transpose(1, 2).unsqueeze(dim=3)
        vectors_matrix = torch.matmul(vectors, vectors.transpose(2, 3))
        bbox = self.bbox_layer(vectors_matrix)
        connections = self.sigmoid(bbox)
        return vectors, connections


class ConnectionsSMPV4(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
    ):
        super(ConnectionsSMPV4, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.connections_classes = connections_classes
        self.bbox_layer = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=connections_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in range(masks.shape[1]):
            #print(i)
            if i != 0:
                cur_mask = masks[:, i, :, :].to(dtype=torch.bool)
                #print(cur_mask.shape)
                #if cur_mask.sum() > 0:
                #print(cur_mask.shape, encoded.shape)
                vector_encoded = []
                for i, cm in enumerate(cur_mask):
                    v_encoded = encoded[:, i, cm]
                    #print(v_encoded.shape)
                    v_encoded = v_encoded.max(dim=1)[0]
                    vector_encoded.append(v_encoded.unsqueeze(0))
                vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
                vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1) # (b,n,f)
        vectors = vectors.transpose(1, 2).unsqueeze(dim=3)
        vectors_matrix = torch.matmul(vectors, vectors.transpose(2, 3))
        bbox = self.bbox_layer(vectors_matrix)
        connections = self.sigmoid(bbox)
        return vectors, connections


class ConnectionsSMPV5(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            connections_classes=4,
    ):
        super(ConnectionsSMPV5, self).__init__()
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
        )

    def forward(self, x, masks):
        encoded = self.model(x)
        vectors = self.extractor(encoded, masks)
        connections = self.connector(vectors, vectors)
        return vectors, connections


class ObjectDetectionSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            #connections_classes=4,
    ):
        super(ObjectDetectionSMP, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.points_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.softmax = torch.nn.Softmax2d()
        self.sigmoid = torch.nn.Sigmoid()
        self.bbox_layer = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.limit = 1000

    def forward(self, x, mask=None):
        encoded = self.model(x)
        points = self.points_classes(encoded)
        points = self.softmax(points)
        points_encode = points.argmax(dim=1)
        if mask is None:
            left_top_points = (points_encode == 1)#.unsqueeze(1)
            right_bottom_points = (points_encode == 2)#.unsqueeze(1)
        else:
            left_top_points = (mask == 1)  # .unsqueeze(1)
            right_bottom_points = (mask == 2)  # .unsqueeze(1)
        out = []
        for b in range(left_top_points.shape[0]):
            #print(left_top_points.shape, encoded.shape)
            left_top_vectors = encoded[b,:,left_top_points[b]][:,:self.limit].unsqueeze(2)
            right_bottom_vectors = encoded[b,:,right_bottom_points[b]][:,:self.limit].unsqueeze(1)
            #print(left_top_vectors.shape, right_bottom_vectors.shape)
            vectors_matrix = torch.matmul(left_top_vectors, right_bottom_vectors).unsqueeze(0)
            vectors_out = self.bbox_layer(vectors_matrix)
            vectors_out = self.sigmoid(vectors_out)
            out.append(vectors_out)
        out = torch.cat(out, dim=0)
        return out, points


class ObjectDetectionSMPV2(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            #connections_classes=4,
    ):
        super(ObjectDetectionSMPV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            activation=None,
            classes=features_size,
        )
        self.points_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.softmax = torch.nn.Softmax2d()
        self.sigmoid = torch.nn.Sigmoid()
        self.bbox_layer = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.limit = 1000
        self.threshold = 0.5

    def forward(self, x, mask=None):
        encoded = self.model(x)
        points = self.points_classes(encoded)
        points = self.sigmoid(points)
        #print(x.shape, encoded.shape,  points.shape)
        if mask is None:
            mask = points
        left_top_points = mask[:, 0, :, :] > self.threshold
        right_bottom_points = mask[:, 1, :, :] > self.threshold
        out = []
        for b in range(left_top_points.shape[0]):
            #print(left_top_points.shape, encoded.shape)
            left_top_vectors = encoded[b,:,left_top_points[b]][:,:self.limit].unsqueeze(2)
            right_bottom_vectors = encoded[b,:,right_bottom_points[b]][:,:self.limit].unsqueeze(1)
            #print(left_top_vectors.shape, right_bottom_vectors.shape)
            vectors_matrix = torch.matmul(left_top_vectors, right_bottom_vectors).unsqueeze(0)
            vectors_out = self.bbox_layer(vectors_matrix)
            vectors_out = self.sigmoid(vectors_out)
            out.append(vectors_out)
        out = torch.cat(out, dim=0)
        return out, points
