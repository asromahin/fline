import torch


class BaseEncoder(torch.nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
