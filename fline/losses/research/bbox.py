import torch

from fline.losses.object_detection.iou import IouDotsLoss


class BboxLoss(torch.nn.Module):
    def __init__(self, device):
        super(BboxLoss, self).__init__()
        self.loss = IouDotsLoss(device)

    def forward(
            self,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor,
    ):
        pred_count = pred_bboxes.shape[1]
        target_count = target_bboxes.shape[1]
        k_keys = list(range(pred_count))
        res_loss = None
        count_loss = 0
        for i in range(target_count):
            target_bbox = target_bboxes[:, i, :]
            corrects_mask = target_bbox.max(dim=1)[0] != 0
            target_bbox = target_bbox[corrects_mask]
            cur_loss = None
            select_k = None
            for k in k_keys:
                pred_bbox = pred_bboxes[:, k, :][corrects_mask]
                #print(target_bbox.shape, pred_bbox.shape, corrects_mask.shape)
                tloss = self.loss(pred_bbox, target_bbox)
                if cur_loss is None:
                    cur_loss = tloss
                    select_k = k
                else:
                    if tloss < cur_loss and tloss >= 0:
                        cur_loss = tloss
                        select_k = k
            if cur_loss is not None:
                k_keys.remove(select_k)
                count_loss += 1
                if res_loss is None:
                    res_loss = cur_loss
                else:
                    res_loss += cur_loss
        for k in k_keys:
            res_loss += 1
            count_loss += 1
        return res_loss/count_loss


class BboxLossFixed(torch.nn.Module):
    def __init__(self, device):
        super(BboxLossFixed, self).__init__()
        self.loss = IouDotsLoss(device)

    def forward(
            self,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor,
    ):
        res_loss = None
        #print('-' * 60)
        #print(pred_bboxes.shape, target_bboxes.shape)
        for i in range(target_bboxes.shape[1]):
            target_bbox = target_bboxes[:, i, :]
            corrects_mask = target_bbox.max(dim=1)[0] != 0
            target_bbox = target_bbox[corrects_mask]
            pred_bbox = pred_bboxes[:, i, :][corrects_mask]
            #print(pred_bbox, target_bbox)
            tloss = self.loss(pred_bbox, target_bbox)
            if res_loss is None:
                res_loss = tloss
            else:
                res_loss += tloss
        return res_loss/target_bboxes.shape[1]


class BboxFeaturesLoss(torch.nn.Module):
    def __init__(self, device):
        super(BboxFeaturesLoss, self).__init__()
        self.loss = IouDotsLoss(device)

    def forward(
            self,
            features: torch.Tensor,
            target_bboxes: torch.Tensor,
    ):
        target_count = len(target_bboxes)
        for i in range(target_count):
            cur_box = target_bboxes[i]
            vector = features