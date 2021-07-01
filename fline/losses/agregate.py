import torch
import typing as tp


class BaseLossAgregator(torch.nn.Module):
    def __init__(self):
        super(BaseLossAgregator, self).__init__()


class AddAgregateLosses(BaseLossAgregator):
    def __init__(self, losses: tp.List[tp.Tuple[tp.Callable, float]]):
        self.losses = losses
        super(AddAgregateLosses, self).__init__()
        
    def forward(self, *args, **kwargs):
        res = None
        for loss, weight in self.losses:
            rloss = loss(*args, **kwargs) * weight
            if res is None:
                res = rloss
            else:
                res += rloss
        return res

