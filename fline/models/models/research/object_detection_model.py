import torch
import segmentation_models_pytorch as smp
from fline.models.models.research.connected_model import SuperModelV4


def iou_loss(pr, gt, device, eps=1e-7):
    #   (b, 4) x,y,w,h,score

    area_pred = pr[:, 2] * pr[:, 3]
    area_gt = gt[:, 2] * gt[:, 3]

    x_min_true = gt[:, 0:1] - gt[:, 2:3] / 2
    x_max_true = gt[:, 0:1] + gt[:, 2:3] / 2
    y_min_true = gt[:, 1:2] - gt[:, 3:4] / 2
    y_max_true = gt[:, 1:2] + gt[:, 3:4] / 2

    x_min_pred = pr[:, 0:1] - pr[:, 2:3] / 2
    x_max_pred = pr[:, 0:1] + pr[:, 2:3] / 2
    y_min_pred = pr[:, 1:2] - pr[:, 3:4] / 2
    y_max_pred = pr[:, 1:2] + pr[:, 3:4] / 2

    #print(torch.cat([x_max_pred, torch.ones(x_max_pred.shape, device=device)], dim=-1).shape)

    x_max_pred = torch.min(torch.cat([x_max_pred, torch.ones(x_max_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    y_max_pred = torch.min(torch.cat([y_max_pred, torch.ones(y_max_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    x_min_pred = torch.max(torch.cat([x_min_pred, torch.zeros(x_min_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    y_min_pred = torch.max(torch.cat([y_min_pred, torch.zeros(y_min_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)

    #print(x_max_pred.shape, y_max_pred.shape, x_min_pred.shape, y_min_pred.shape)
    #print(x_max_true.shape, y_max_true.shape, x_min_true.shape, y_min_true.shape)

    overlap_0 = torch.max(torch.cat([x_min_true, x_min_pred], dim=-1), dim=-1)[0]
    overlap_1 = torch.max(torch.cat([y_min_true, y_min_pred], dim=-1), dim=-1)[0]
    overlap_2 = torch.min(torch.cat([x_max_true, x_max_pred], dim=-1), dim=-1)[0]
    overlap_3 = torch.min(torch.cat([y_max_true, y_max_pred], dim=-1), dim=-1)[0]

    if overlap_2 < overlap_0 or overlap_3 < overlap_1:
        #print('mse')
        return 1+torch.nn.functional.mse_loss(pr, gt)

    # intersection area
    intersection = ((overlap_2 - overlap_0) * (overlap_3 - overlap_1)).sum()
    res = (intersection / (area_pred + area_gt - intersection + eps)).mean()
    #print('iou')
    return 1-res


def find_intersection(set_1, set_2, device):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = upper_bounds-lower_bounds
    #print(intersection_dims.shape, upper_bounds.shape)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2, device):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2, device)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return 1-intersection / union  # (n1, n2)


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)



class BboxLoss(torch.nn.Module):
    def __init__(self, device):
        super(BboxLoss, self).__init__()
        self.loss = iou_loss
        self.device = device

    def forward(
            self,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor,
    ):
        #(b, n, 4) (b, k, 4)
        #print(pred_bboxes.shape, target_bboxes.shape)
        pred_count = pred_bboxes.shape[1]
        target_count = target_bboxes.shape[1]
        k_keys = list(range(pred_count))
        res_loss = None
        for i in range(target_count):
            target_bbox = target_bboxes[:, i, :]
            cur_loss = None
            select_k = None
            # if len(k_keys) > 0:
            for k in k_keys:
                pred_bbox = pred_bboxes[:, k, :]
                tloss = self.loss(pred_bbox, target_bbox, self.device)
                #print(tloss)
                if cur_loss is None:
                    cur_loss = tloss
                    select_k = k
                else:
                    if tloss < cur_loss and tloss >= 0:
                        cur_loss = tloss
                        select_k = k
            if cur_loss is not None:
                k_keys.remove(select_k)
                if res_loss is None:
                    res_loss = cur_loss
                else:
                    res_loss += cur_loss
            # else:
            #     cur_loss = torch.ones((target_bbox.shape[0]), device=self.device)
            #     if res_loss is None:
            #         res_loss = cur_loss
            #     else:
            #         res_loss += cur_loss

        return res_loss/target_count


class BboxLossV2(torch.nn.Module):

    def __init__(self, device):
        super(BboxLossV2, self).__init__()
        self.device = device

    def forward(
            self,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor,
    ):
        res_loss = None
        for i in range(pred_bboxes.shape[0]):
            overlap_loss = find_jaccard_overlap(target_bboxes[i], pred_bboxes[i], self.device)
            batch_loss = None
            for j in range(overlap_loss.shape[0]):
                #print(overlap_loss[j])
                #print(torch.min(overlap_loss[j]))
                cur_loss = torch.min(overlap_loss[j])
                if batch_loss is None:
                    batch_loss = cur_loss
                else:
                    batch_loss += cur_loss
            if res_loss is None:
                res_loss = batch_loss
            else:
                res_loss += batch_loss
        return res_loss/target_bboxes.shape[1]


class ObjectDetectionModel(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(ObjectDetectionModel, self).__init__()
    self.backbone_name = backbone_name
    self.model = smp.FPN(self.backbone_name, classes=last_features)
    self.connected = conv3x3(last_features, connected_channels)
    self.bbox_layer = torch.nn.Linear(last_features, 4)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    encoded = self.model(x)
    source_connected_out = self.connected(encoded)
    source_connected_out = torch.softmax(source_connected_out, dim=1)
    connected_out = source_connected_out.argmax(dim=1)
    out_bboxes = []
    zero_bbox = []
    for i in torch.unique(connected_out):
            cur_mask = (connected_out == i).to(dtype=torch.float32)
            cur_encoded = encoded * cur_mask
            cur_encoded = torch.max(cur_encoded, dim=3)[0]
            cur_encoded = torch.max(cur_encoded, dim=2)[0]

            bbox = self.bbox_layer(cur_encoded)
            bbox = bbox.unsqueeze(1)
            bbox = self.sigmoid(bbox)
            # print(bbox.shape)
            # print(bbox)
            if i != 0:
                out_bboxes.append(bbox)
            else:
                zero_bbox.append(bbox)
    if len(out_bboxes) == 0:
        out_bboxes = zero_bbox
    out_bboxes = torch.cat(out_bboxes, dim=1)
    return out_bboxes, source_connected_out


class ObjectDetectionModelV2(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(ObjectDetectionModelV2, self).__init__()
    self.backbone_name = backbone_name
    self.model = smp.FPN(self.backbone_name, classes=last_features)
    self.connected = conv3x3(last_features, connected_channels)
    self.bbox_layer = torch.nn.Linear(last_features, last_features)
    self.bbox_layer_out = torch.nn.Linear(last_features, 4)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    encoded = self.model(x)
    source_connected_out = self.connected(encoded)
    source_connected_out = torch.softmax(source_connected_out, dim=1)
    connected_out = source_connected_out.argmax(dim=1)
    out_bboxes = []
    zero_bbox = []
    for i in torch.unique(connected_out):
            cur_mask = (connected_out == i).to(dtype=torch.float32)
            cur_encoded = encoded * cur_mask
            cur_encoded = torch.max(cur_encoded, dim=3)[0]
            cur_encoded = torch.max(cur_encoded, dim=2)[0]

            bbox = self.bbox_layer(cur_encoded)
            bbox = self.bbox_layer_out(bbox)
            bbox = bbox.unsqueeze(1)
            bbox = self.sigmoid(bbox)
            # print(bbox.shape)
            # print(bbox)
            if i != 0:
                out_bboxes.append(bbox)
            else:
                zero_bbox.append(bbox)
    if len(out_bboxes) == 0:
        out_bboxes = zero_bbox
    out_bboxes = torch.cat(out_bboxes, dim=1)
    return out_bboxes, source_connected_out


class ObjectDetectionModelV3(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(ObjectDetectionModelV3, self).__init__()
    self.backbone_name = backbone_name
    self.model = SuperModelV4(
        backbone_name=backbone_name,
        last_features=last_features,
        connected_channels=connected_channels,
    )
    self.linear_channels = sum(self.model.features_channels[:-1])
    self.bbox_layer = torch.nn.Linear(self.linear_channels, self.linear_channels//2)
    self.bbox_layer_out = torch.nn.Linear(self.linear_channels//2, 4)
    self.sigmoid = torch.nn.Sigmoid()
    self.pool = torch.nn.MaxPool2d((2, 2))

  def forward(self, x):
    encoded, encoded_features = self.model(x)
    connected_out = encoded.argmax(dim=1)
    out_bboxes = []
    zero_bbox = []
    for i in torch.unique(connected_out):
        cur_mask = (connected_out == i).to(dtype=torch.float32)
        vector_encoded = None
        for j, feature in enumerate(encoded_features[::-1]):
            cur_encoded = feature * cur_mask
            cur_encoded = torch.max(cur_encoded, dim=3)[0]
            cur_encoded = torch.max(cur_encoded, dim=2)[0]
            #print(cur_encoded.shape)
            if vector_encoded is None:
                vector_encoded = cur_encoded
            else:
                vector_encoded = torch.cat([vector_encoded, cur_encoded], dim=1)
            cur_mask = self.pool(cur_mask)
        #print(vector_encoded.shape, self.linear_channels)
        bbox = self.bbox_layer(vector_encoded)
        bbox = self.bbox_layer_out(bbox)
        bbox = bbox.unsqueeze(1)
        bbox = self.sigmoid(bbox)
        # print(bbox.shape)
        # print(bbox)
        if i != 0:
            out_bboxes.append(bbox)
        else:
            zero_bbox.append(bbox)
    if len(out_bboxes) == 0:
        out_bboxes = zero_bbox
    out_bboxes = torch.cat(out_bboxes, dim=1)
    return out_bboxes, encoded

