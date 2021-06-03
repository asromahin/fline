import torch
import segmentation_models_pytorch as smp

from fline.models.models.segmentation.fpn import TimmFPN


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
            classes=1,
            features_block=None,
    ):
        super(ObjectDetectionModelV2, self).__init__()
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            classes=4+classes,
            return_dict=True,
            features_block=features_block,
        )
        self.bbox_layer = torch.nn.Linear(self.model.out_channels, self.model.out_channels//2)
        self.bbox_layer_out = torch.nn.Linear(self.model.out_channels//2, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        res_dict = self.model(x)
        encoded = res_dict['out']
        encoded_classes = res_dict['classes']
        connected_out = encoded_classes.argmax(dim=1)
        out_bboxes = []
        zero_bbox = []
        for i in torch.unique(connected_out):
            cur_mask = (connected_out == i).to(dtype=torch.float32)
            vector_encoded = cur_mask * encoded
            vector_encoded = vector_encoded.max(dim=2)[0]
            vector_encoded = vector_encoded.max(dim=2)[0]
            bbox = self.bbox_layer(vector_encoded)
            bbox = self.bbox_layer_out(bbox)
            bbox = bbox.unsqueeze(1)
            bbox = self.sigmoid(bbox)
            if i != 0:
                out_bboxes.append(bbox)
            else:
                zero_bbox.append(bbox)
        if len(out_bboxes) == 0:
            out_bboxes = zero_bbox
        out_bboxes = torch.cat(out_bboxes, dim=1)
        return out_bboxes, encoded_classes
