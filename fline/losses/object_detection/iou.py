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
