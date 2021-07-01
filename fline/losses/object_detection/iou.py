import torch

from fline.metrics.object_detection.iou import iou_dots


def iou_dots_loss(pr, gt, device, eps=1e-7):
    return 1 - iou_dots(pr, gt, device, eps)


class IouDotsLoss(torch.nn.Module):
    def __init__(self, device):
        super(IouDotsLoss, self).__init__()
        self.device = device

    def forward(self, pr, gt):
        return iou_dots_loss(pr, gt, self.device)


class IouDotsLossAll(torch.nn.Module):
    def __init__(self, device):
        super(IouDotsLossAll, self).__init__()
        self.device = device

    def forward(self, pr, gt):
        pr = torch.reshape(pr, (pr.shape[0] * pr.shape[1], *pr.shape[2:]))
        gt = torch.reshape(gt, (gt.shape[0] * gt.shape[1], *gt.shape[2:]))
        return iou_dots_loss(pr, gt, self.device)
