import torch

from fline.losses.segmentation.dice import BCEDiceLoss


class ConnectLoss(torch.nn.Module):

    def __init__(self, device):
        super(ConnectLoss, self).__init__()
        self.loss = BCEDiceLoss(activation=None)
        self.device = device

    def forward(
            self,
            pred_instance_mask: torch.Tensor,
            target_mask: torch.Tensor,
    ):
        res_loss = self.loss(
            pred_instance_mask[:, 0:1, :, :].to(torch.float32),
            (target_mask == 0).to(torch.float32),
        )
        k_keys = list(range(pred_instance_mask.shape[1]))
        k_keys.remove(0)
        n_keys = list(torch.unique(target_mask))
        n_keys.remove(0)
        for n in n_keys:
            target_cur_mask = (target_mask == n)
            cur_loss = None
            select_k = None
            for k in k_keys:
                pred_cur_mask = pred_instance_mask[:, k:k+1, :, :]
                tloss = self.loss(pred_cur_mask.to(torch.float32), target_cur_mask.to(torch.float32))
                if cur_loss is None:
                    cur_loss = tloss
                    select_k = k
                else:
                    if tloss < cur_loss:
                        cur_loss = tloss
                        select_k = k
            if cur_loss is not None:
                k_keys.remove(select_k)
                res_loss += cur_loss
        return res_loss/len(n_keys)
