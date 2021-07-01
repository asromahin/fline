import torch
import segmentation_models_pytorch as smp

from fline.models.models.research.connected_components import ConnectedComponents
from fline.losses.segmentation.dice import BCEDiceLoss
from fline.models.blocks.attention import ConnCompAttention
from fline.models.models.segmentation.fpn import TimmFPN


class ConnectedModelSMP(torch.nn.Module):
    def __init__(
            self,
            backbone_name='resnet18',
            classes=2,
            activation='softmax',
            device='cpu',
    ):
        super(ConnectedModelSMP, self).__init__()
        # self.model = TimmFPN(
        #     backbone_name=backbone_name,
        #     #features_block='each',
        #     classes=classes,
        #     activation=None,
        # )
        self.model = smp.FPN(encoder_name=backbone_name, classes=classes, activation=None)
        self.activation = torch.nn.Softmax2d()

    def forward(self, x):
        out = self.model(x)
        out = self.activation(out)
        return out


class ConnectedLoss(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLoss, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = BCEDiceLoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        res_loss = self.loss((pred_out[:,1:2,:,:]*(pred_masks.unsqueeze(dim=1) > 0)).to(dtype=torch.float32), (target_mask > 0).to(dtype=torch.float32))
        pred_uniq = list(torch.unique(pred_masks))
        target_uniq = list(torch.unique(target_mask))
        #print(pred_uniq, len(pred_uniq))
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                #print(v, cur_mask.sum()/cur_mask.shape[1]/cur_mask.shape[2])
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for t in target_uniq:
                    min_loss = None
                    min_ind = None
                    target_m = (target_mask == t).to(dtype=torch.float32)
                    for f in cur_uniq:
                        c_mask = pred_out[:, v:v+1, :, :] * (cur_mask == f)
                        #print(c_mask.dtype, target_m.dtype)
                        cur_loss = self.loss(c_mask, target_m)
                        if min_loss is None:
                            min_loss = cur_loss
                            min_ind = f
                        else:
                            if cur_loss < min_loss:
                                min_loss = cur_loss
                                min_ind = f
                    if min_loss is not None:
                        if res_loss is None:
                            res_loss = min_loss
                        else:
                            res_loss += min_loss
                        cur_uniq.remove(min_ind)
                        target_uniq.remove(t)
                res_loss += len(cur_uniq)
        res_loss += len(target_uniq)
        return res_loss


class ConnectedLossV2(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLossV2, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = BCEDiceLoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        res_loss = self.loss((pred_out[:,1:2,:,:]*(pred_masks.unsqueeze(dim=1) > 0)).to(dtype=torch.float32), (target_mask > 0).to(dtype=torch.float32))
        pred_masks = pred_out.argmax(dim=1)
        pred_uniq = list(torch.unique(pred_masks))
        pred_placeholder = torch.zeros_like(pred_masks, dtype=torch.float32)
        target_uniq = list(torch.unique(target_mask))
        if 0 in target_uniq:
            target_uniq.remove(0)
        last_i = 1
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for c in cur_uniq:
                    if c != 0:
                        c_mask = (cur_mask == c)
                        pred_placeholder += (c_mask*c + last_i)
                last_i += len(cur_uniq)
        cur_uniq = list(torch.unique(pred_placeholder))
        if 0 in cur_uniq:
            cur_uniq.remove(0)
        #print(len(cur_uniq))
        #print(len(cur_uniq))
        for t in target_uniq:
            min_loss = None
            min_ind = None
            target_m = (target_mask == t).to(dtype=torch.float32)
            for f in cur_uniq:
                c_mask = (pred_placeholder == f).to(dtype=torch.float32).unsqueeze(dim=1)
                # print(c_mask.dtype, target_m.dtype)
                cur_loss = self.loss(c_mask, target_m)
                if min_loss is None:
                    min_loss = cur_loss
                    min_ind = f
                else:
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        min_ind = f
            if min_loss is not None:
                if res_loss is None:
                    res_loss = min_loss
                else:
                    res_loss += min_loss
                cur_uniq.remove(min_ind)
                target_uniq.remove(t)
        res_loss += len(cur_uniq)
        res_loss += len(target_uniq)
        return res_loss


class ConnectedLossV3(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLossV3, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = BCEDiceLoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        res_loss = self.loss((pred_out[:,1:2,:,:]*(pred_masks.unsqueeze(dim=1) > 0)).to(dtype=torch.float32), (target_mask > 0).to(dtype=torch.float32))
        pred_masks = pred_out.argmax(dim=1)
        pred_uniq = list(torch.unique(pred_masks))
        pred_placeholder = torch.zeros_like(pred_masks, dtype=torch.float32)
        target_uniq = list(torch.unique(target_mask))
        if 0 in target_uniq:
            target_uniq.remove(0)
        last_i = 1
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for c in cur_uniq:
                    if c != 0:
                        c_mask = (cur_mask == c)
                        pred_placeholder += (c_mask*c + last_i)
                last_i += len(cur_uniq)

        #print(len(cur_uniq))
        #print(len(cur_uniq))
        multiply_mask = pred_placeholder*target_mask
        for t in target_uniq:
            min_loss = None
            min_ind = None
            target_m = (target_mask == t).to(dtype=torch.float32)
            cur_mult_mask = multiply_mask * target_mask
            cur_uniq = list(torch.unique(cur_mult_mask))
            if 0 in cur_uniq:
                cur_uniq.remove(0)
            for f in cur_uniq:
                c_mask = (cur_mult_mask == f).to(dtype=torch.float32)#.unsqueeze(dim=1)
                # print(c_mask.dtype, target_m.dtype)
                cur_loss = self.loss(c_mask, target_m)
                if min_loss is None:
                    min_loss = cur_loss
                    min_ind = f
                else:
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        min_ind = f
            if min_loss is not None:
                if res_loss is None:
                    res_loss = min_loss
                else:
                    res_loss += min_loss
                cur_uniq.remove(min_ind)
                target_uniq.remove(t)
            res_loss += len(cur_uniq)
        res_loss += len(target_uniq)
        return res_loss/len(torch.unique(target_mask))


class ConnectedLossV4(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLossV4, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = BCEDiceLoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        res_loss = self.loss((pred_out[:,0:1,:,:]*(pred_masks.unsqueeze(dim=1) == 0)).to(dtype=torch.float32), (target_mask == 0).to(dtype=torch.float32))
        pred_masks = pred_out.argmax(dim=1)
        pred_uniq = list(torch.unique(pred_masks))
        pred_placeholder = torch.zeros_like(pred_masks, dtype=torch.float32)
        target_uniq = list(torch.unique(target_mask))
        if 0 in target_uniq:
            target_uniq.remove(0)
        last_i = 1
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for c in cur_uniq:
                    if c != 0:
                        c_mask = (cur_mask == c)
                        pred_placeholder += (c_mask*c + last_i)
                last_i += len(cur_uniq)


        #multiply_mask = pred_placeholder*target_mask
        for t in target_uniq:
            target_m = (target_mask == t)[:, 0]
            #print(pred_placeholder.shape, target_m.shape)
            med = torch.median(pred_placeholder[target_m])
            full_med_mask = (pred_placeholder == med).to(dtype=torch.float32)
            res_loss += self.loss(full_med_mask, target_m.to(dtype=torch.float32))
        return res_loss/(len(torch.unique(target_mask))+1)


class ConnectedLossV5(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLossV5, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = torch.nn.BCELoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        res_loss = self.loss(
            (pred_out[:, 0:1, :, :]*(pred_masks.unsqueeze(dim=1) == 0)).to(dtype=torch.float32),
            (target_mask == 0).to(dtype=torch.float32),
        )
        pred_uniq = list(torch.unique(pred_masks))
        pred_placeholder = torch.zeros_like(pred_masks, dtype=torch.float32, device=self.device)
        pred_placeholder_out = torch.zeros_like(pred_masks, dtype=torch.float32, device=self.device)
        target_uniq = list(torch.unique(target_mask))
        if 0 in target_uniq:
            target_uniq.remove(0)
        last_i = 1
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for c in cur_uniq:
                    if c != 0:
                        c_mask = (cur_mask == c)
                        pred_placeholder = (c_mask + last_i)
                        #print(c_mask.shape, pred_out.shape, pred_placeholder_out.shape)
                        pred_placeholder_out[c_mask] = pred_out[:, v][c_mask]
                last_i += len(cur_uniq)


        #multiply_mask = pred_placeholder*target_mask
        for t in target_uniq:
            target_m = (target_mask == t)[:, 0]
            #print(pred_placeholder.shape, target_m.shape)
            values = pred_placeholder[target_m]
            med = torch.median(values)

            full_med_mask = (pred_placeholder == med)#.to(dtype=torch.float32)
            #print(pred_placeholder_out.mean().item(), (pred_placeholder_out * full_med_mask).mean().item())
            res_loss += self.loss(pred_placeholder_out * full_med_mask, target_m.to(dtype=torch.float32))
            extra_mask = ((pred_placeholder != med) * target_m)#.to(dtype=torch.float32)
            res_loss += (pred_placeholder_out*extra_mask).sum()/target_m.sum()
        return res_loss/(len(torch.unique(target_mask))*2+1)


class ConnectedLossV6(torch.nn.Module):
    def __init__(self, device):
        super(ConnectedLossV6, self).__init__()
        self.conn_comp = ConnectedComponents()
        self.loss = BCEDiceLoss()
        self.device = device

    def forward(self, pred_out, target_mask):

        pred_masks = pred_out.argmax(dim=1)
        zero_mask = pred_masks.max(dim=1)[0]
        res_loss = self.loss(((zero_mask == 0)).to(dtype=torch.float32), (target_mask == 0).to(dtype=torch.float32))
        #pred_masks = pred_out.argmax(dim=1)
        pred_uniq = list(torch.unique(pred_masks))
        pred_placeholder = torch.zeros_like(pred_masks, dtype=torch.float32, device=self.device)
        pred_placeholder_out = torch.zeros_like(pred_masks, dtype=torch.float32, device=self.device)
        target_uniq = list(torch.unique(target_mask))
        if 0 in target_uniq:
            target_uniq.remove(0)
        last_i = 1
        for v in pred_uniq:
            if v != 0:
                cur_mask = (pred_masks == v)
                cur_mask = self.conn_comp(cur_mask)
                cur_uniq = list(torch.unique(cur_mask))
                for c in cur_uniq:
                    if c != 0:
                        c_mask = (cur_mask == c)
                        pred_placeholder += (c_mask + last_i)
                        #print(c_mask.shape, pred_out.shape, pred_placeholder_out.shape)
                        pred_placeholder_out[c_mask] = pred_out[:, v][c_mask]
                last_i += len(cur_uniq)


        #multiply_mask = pred_placeholder*target_mask
        for t in target_uniq:
            target_m = (target_mask == t)[:, 0]
            #print(pred_placeholder.shape, target_m.shape)
            values = pred_placeholder[target_m]
            med = torch.median(values)
            full_med_mask = (pred_placeholder == med)#.to(dtype=torch.float32)
            res_loss += self.loss(pred_placeholder_out * full_med_mask, target_m.to(dtype=torch.float32))
            extra_mask = ((pred_placeholder != med) * target_m)#.to(dtype=torch.float32)
            res_loss += (pred_placeholder_out*extra_mask).sum()/target_m.sum()
        return res_loss/(len(torch.unique(target_mask))*2+1)
