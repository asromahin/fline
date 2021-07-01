import torch

from fline.utils.logging.base_group import BaseGroup


class ConcatImage(BaseGroup):
    def __init__(self, log_type, keys, dim):
        super(ConcatImage, self).__init__(log_type)
        self.keys = keys
        self.dim = dim

    def __call__(self, data):
        res = None
        for k in self.keys:
            d = data[k]
            if res is None:
                res = d
            else:
                res = torch.cat([res, d], dim=self.dim)
        return res


class ProcessImage(BaseGroup):
    def __init__(self, log_type, process_func):
        super(ProcessImage, self).__init__(log_type)
        self.process_func = process_func

    def __call__(self, data):
        return self.process_func(data)
