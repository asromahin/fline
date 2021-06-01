import torch

from fline.metrics.segmentation.segmentation import f_score


class DiceLoss(torch.nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCELoss(torch.nn.Module):
    __name__ = 'bce_loss'

    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        return bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = BCELoss()

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce