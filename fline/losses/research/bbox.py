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
        for i in range(target_count):
            target_bbox = target_bboxes[:, i, :]
            cur_loss = None
            select_k = None
            for k in k_keys:
                pred_bbox = pred_bboxes[:, k, :]
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
                if res_loss is None:
                    res_loss = cur_loss
                else:
                    res_loss += cur_loss
        return res_loss/target_count