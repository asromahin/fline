import torch


def iou_dots(pr, gt, device, eps=1e-7):
    #   (batch_size, 4) x_center, y_center, w, h,score

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

    x_max_pred = torch.min(torch.cat([x_max_pred, torch.ones(x_max_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    y_max_pred = torch.min(torch.cat([y_max_pred, torch.ones(y_max_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    x_min_pred = torch.max(torch.cat([x_min_pred, torch.zeros(x_min_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)
    y_min_pred = torch.max(torch.cat([y_min_pred, torch.zeros(y_min_pred.shape, device=device)], dim=-1), dim=-1)[0].unsqueeze(-1)

    overlap_0 = torch.max(torch.cat([x_min_true, x_min_pred], dim=-1), dim=-1)[0]
    overlap_1 = torch.max(torch.cat([y_min_true, y_min_pred], dim=-1), dim=-1)[0]
    overlap_2 = torch.min(torch.cat([x_max_true, x_max_pred], dim=-1), dim=-1)[0]
    overlap_3 = torch.min(torch.cat([y_max_true, y_max_pred], dim=-1), dim=-1)[0]

    # возвращаем разницу между векторами, если прямоугольники не пересекаются
    if overlap_2 < overlap_0 or overlap_3 < overlap_1:
        return -torch.nn.functional.mse_loss(pr, gt)

    # intersection area
    intersection = ((overlap_2 - overlap_0) * (overlap_3 - overlap_1)).sum()
    return (intersection / (area_pred + area_gt - intersection + eps)).mean()


class IouDots(torch.nn.Module):
    def __init__(self, device):
        self.metric = iou_dots
        self.device = device

    def forward(self, pr, gt):
        return self.metric(pr, gt, device=self.device)

