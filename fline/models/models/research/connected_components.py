import torch
from torch.nn.functional import max_pool2d


class ConnectedComponents(torch.nn.Module):
    def __init__(self):
        super(ConnectedComponents, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # x.shape = [batch_size, channels, height, width]
        if len(x.size()) == 3:
            batch_size, height, width = x.size()
        else:
            batch_size, channels, height, width = x.size()
        mult = height * width
        mask = (mult-torch.arange(mult, device=x.device)).reshape((height, width))
        mask = x * mask
        mask = mask.to(dtype=torch.float32)
        #torch.max(mask, keepdim=True, out=mask)
        last_len = 0
        cur_len = -1
        #print(torch.unique(x))s
        #print(x.max())
        #print(mask.shape, x.shape, (x==1).sum())
        while cur_len != last_len:
            mask[x == 1] = self.pool(mask)[x == 1]
            #print(cur_len)
            last_len = cur_len
            cur_len =+ torch.unique(mask).shape[0]
        return mask