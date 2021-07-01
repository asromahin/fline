import torch
import segmentation_models_pytorch as smp

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.encoders.timm import TimmEncoder
from fline.models.models.research.extractor import VectorsFromMask, VectorsFromMaskV2
from fline.models.models.research.connect_net import ConnectNet
from fline.models.models.research.connected_components import ConnectedComponents
from fline.losses.object_detection.iou import IouDotsLoss



class MapNetTimm(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            #features_size=64,
            classes=1,
            device='cpu',
    ):
        super(MapNetTimm, self).__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.model = TimmFPN(
            backbone_name=backbone_name,
            #activation=None,
            #classes=features_size,
        )
        features_size = self.model.out_feature_channels
        self.out_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_bbox = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x, left_top_points=None, right_bottom_points=None):
        encoded = self.model(x)
        points = self.out_classes(encoded)
        points = self.softmax(points)
        bboxes = self.out_bbox(encoded)
        pos_mask = make_pos_mask(encoded.shape[-2:], self.device)
        bboxes[:,:2,:,:] += pos_mask
        return points, bboxes


class MapNetSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            features_size=64,
            classes=1,
            device='cpu',
    ):
        super(MapNetSMP, self).__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.model = smp.FPN(
            encoder_name=backbone_name,
            #activation=None,
            classes=features_size,
        )
        #features_size = self.model.out_feature_channels
        self.out_classes = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_bbox = torch.nn.Conv2d(
            in_channels=features_size,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x, left_top_points=None, right_bottom_points=None):
        encoded = self.model(x)
        points = self.out_classes(encoded)
        points = self.softmax(points)
        bboxes = self.out_bbox(encoded)
        pos_mask = make_pos_mask(encoded.shape[-2:], self.device)
        bboxes[:,:2,:,:] += pos_mask
        return points, bboxes


def make_pos_mask(shape, device):
    res = torch.arange(shape[0]*shape[1], device=device).reshape(shape)
    x = ((res % shape[1])/shape[1]).unsqueeze(dim=0)
    y = ((res / shape[1])/shape[0]).unsqueeze(dim=0)
    res_mask = torch.cat([x, y], dim=0)
    return res_mask


# class MapLoss(torch.nn.Module):
#     def __init__(self, device):
#         super(MapLoss, self).__init__()
#         self.loss = IouDotsLoss(device)
#         self.device = device
#         self.mse = torch.nn.MSELoss()
#         self.limit = 32
#
#     def forward(self, points, bboxes, target_bboxes):
#         res_loss = None
#         for b in range(points.shape[0]):
#             mask = points[b].argmax(dim=0) == 1
#             ph = torch.zeros(points[b].shape, device=self.device)
#             ph[0, :, :] = 1
#             #if res_loss is None:
#             #    res_loss = self.mse(points[b], ph)
#             #else:
#             #    res_loss += self.mse(points[b], ph)
#             #print(points.shape, mask.shape, bboxes.shape)
#             pred_bboxes = bboxes[b, :, mask]
#             #print(pred_bboxes.shape)
#             pred_bboxes = pred_bboxes.transpose(0, 1)
#             for i, target_bbox in enumerate(target_bboxes[b]):
#                 target_loss = None
#                 target_ind = None
#                 # add_loss = len(pred_bboxes) - self.limit
#                 dif_bboxes = torch.abs(pred_bboxes[:,:2] - target_bbox[:2])
#
#                 pred_bboxes = pred_bboxes[:self.limit]
#                 pred_bboxes_ind = list(range(len(pred_bboxes)))
#                 for j in pred_bboxes_ind:
#                     box = pred_bboxes[j]
#                     #print(box.shape, target_bbox.shape)
#                     cur_loss = self.loss(box.unsqueeze(dim=0), target_bbox.unsqueeze(dim=0))
#                     if target_loss is None:
#                         target_loss = cur_loss
#                         target_ind = j
#                     elif cur_loss < target_loss:
#                         target_loss = cur_loss
#                         target_ind = j
#                 if target_loss is not None:
#                     if res_loss is None:
#                         res_loss = target_loss
#                     else:
#                         res_loss += target_loss
#                     pred_bboxes_ind.remove(target_ind)
#                 else:
#                     res_loss += 1
#             #for i in pred_bboxes_ind:
#             res_loss += len(pred_bboxes_ind)
#             #res_loss += add_loss
#         return res_loss/points.shape[0]


class MapLoss(torch.nn.Module):
    def __init__(self, device):
        super(MapLoss, self).__init__()
        self.loss = IouDotsLoss(device)
        self.device = device
        self.mse = torch.nn.CrossEntropyLoss()
        self.limit = 32

    def forward(self, points, bboxes, target_bboxes):
        res_loss = None
        for b in range(points.shape[0]):
            mask = points[b].argmax(dim=0) == 1

            for i, target_bbox in enumerate(target_bboxes[b]):
                x1 = int((target_bbox[0] - target_bbox[2] / 2) * mask.shape[1])
                x2 = int((target_bbox[0] + target_bbox[2] / 2) * mask.shape[1])
                y1 = int((target_bbox[1] - target_bbox[3] / 2) * mask.shape[0])
                y2 = int((target_bbox[1] + target_bbox[3] / 2) * mask.shape[0])
                cmask = torch.zeros_like(points[b:b+1], dtype=torch.bool, device=self.device)
                cmask[b, 1, y1:y2, x1:x2] = 1
                cmask[:, 0] = ~cmask[:, 1]
                # closs = self.mse(points[b:b+1], cmask.to(dtype=torch.float32).argmax(dim=1))
                # if res_loss is None:
                #     res_loss = closs
                # else:
                #     res_loss += closs
                kmask = cmask[0, 1] * mask
                # print(mask.shape, cmask.shape)
                pred_bboxes = bboxes[b, :, kmask]
                pred_bboxes = pred_bboxes.transpose(0, 1)
                cur_iou = bbox_iou(target_bbox, pred_bboxes)
                if len(cur_iou) > 0:
                    cur_loss = 1-cur_iou.mean()
                else:
                    cur_loss = self.mse(points[b:b + 1], cmask.to(dtype=torch.float32).argmax(dim=1))

                if res_loss is None:
                    res_loss = cur_loss
                else:
                    res_loss += cur_loss


            # for i in pred_bboxes_ind:
            #res_loss += len(pred_bboxes_ind)
            # res_loss += add_loss
        return res_loss / points.shape[0]

#
# class MapLoss(torch.nn.Module):
#     def __init__(self, device):
#         super(MapLoss, self).__init__()
#         self.loss = IouDotsLoss(device)
#         self.device = device
#         self.mse = torch.nn.CrossEntropyLoss()
#         #self.limit = 1000
#
#     def forward(self, points, bboxes, target_bboxes):
#         res_loss = None
#         for b in range(points.shape[0]):
#             mask = points[b].argmax(dim=0) == 1
#             pred_bboxes = bboxes[b, :, mask]
#             pred_bboxes = pred_bboxes.transpose(0, 1)
#             for box in target_bboxes[b]:
#                 #print(box.shape, pred_bboxes.shape)
#                 cur_loss = 1-bbox_iou(box, pred_bboxes).max()
#                 if res_loss is None:
#                     res_loss = cur_loss
#                 else:
#                     res_loss += cur_loss
#                 if cur_loss is None:
#                     res_loss += 1
#         return res_loss / points.shape[0]


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU