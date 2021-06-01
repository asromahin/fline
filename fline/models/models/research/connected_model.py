import torch
import timm
import segmentation_models_pytorch as smp


class ConnectLoss(torch.nn.Module):

    def __init__(self, device):
        super(ConnectLoss, self).__init__()
        self.loss = BCEDiceLoss()
        self.mse = torch.nn.MSELoss()
        self.device = device

    def forward(
            self,
            pred_instance_mask: torch.Tensor,
            pred_score: torch.Tensor,
            cls_out: torch.Tensor,
            target_mask: torch.Tensor,
    ):
        res_loss = self.mse(pred_score, torch.zeros(pred_score.size(), device=self.device))
        res_loss += self.loss(cls_out, (target_mask > 0).to(dtype=torch.float32))
        k_keys = torch.unique(pred_instance_mask)
        n_keys = torch.unique(target_mask)
        for n in n_keys:
            target_cur_mask = (target_mask == n)
            cur_loss = None
            for k in k_keys:
                pred_cur_mask = (pred_instance_mask == k)
                tloss = self.loss(pred_cur_mask.to(torch.float32), target_cur_mask.to(torch.float32))
                if cur_loss is None:
                    cur_loss = tloss
                else:
                    if tloss < cur_loss:
                        cur_loss = tloss
            if cur_loss is not None:
                res_loss += cur_loss
        return res_loss/len(n_keys)


class ConnectLossV2(torch.nn.Module):

    def __init__(self, device):
        super(ConnectLossV2, self).__init__()
        self.loss = BCEDiceLoss(activation=None)
        self.device = device

    def forward(
            self,
            pred_instance_mask: torch.Tensor,
            cls_out: torch.Tensor,
            target_mask: torch.Tensor,
    ):
        res_loss = self.loss(cls_out, (target_mask > 0).to(dtype=torch.float32))
        res_loss += self.loss(
            pred_instance_mask[:, 0:1, :, :].to(torch.float32),
            (target_mask == 0).to(torch.float32),
        )
        all_k_keys = list(range(pred_instance_mask.shape[1]))
        all_k_keys.remove(0)
        n_keys = list(torch.unique(target_mask))
        n_keys.remove(0)
        for n in n_keys:
            target_cur_mask = (target_mask == n)
            cur_loss = None
            select_k = None
            k_keys = all_k_keys
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
                all_k_keys.remove(select_k)
                res_loss += cur_loss
        return res_loss/len(n_keys)


class ConnectLossV3(torch.nn.Module):

    def __init__(self, device):
        super(ConnectLossV3, self).__init__()
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


class SuperModel(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(SuperModel, self).__init__()
    self.backbone_name = backbone_name
    self.encoder = timm.create_model(backbone_name, features_only=True)
    self.features_channels = [last_features, *self.get_features()]
    self.decoder = Decoder(features=self.features_channels)
    #self.out_conv = conv3x3(last_features, last_features)
    self.classify = conv3x3(last_features, classes)
    self.connected = conv3x3(last_features, connected_channels)

  def get_features(self):
    with torch.no_grad():
      features = self.encoder(torch.zeros((1, 3, 64, 64)))
    return [f.shape[1] for f in features]

  def forward(self, x):
    features = self.encoder(x)
    encoded = self.decoder(features)
    #output = self.out_conv(encoded)
    cls_out = self.classify(encoded)
    connected_out = self.connected(encoded)
    x_long = connected_out.to(dtype=torch.long)
    x_score = (connected_out - x_long)**2
    return x_long, x_score, cls_out


class SuperModelV2(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(SuperModelV2, self).__init__()
    self.backbone_name = backbone_name
    self.encoder = timm.create_model(backbone_name, features_only=True)
    self.features_channels = [last_features, *self.get_features()]
    self.decoder = Decoder(features=self.features_channels)
    #self.out_conv = conv3x3(last_features, last_features)
    self.classify = conv3x3(last_features, classes)
    self.connected = conv3x3(last_features, connected_channels)

  def get_features(self):
    with torch.no_grad():
      features = self.encoder(torch.zeros((1, 3, 64, 64)))
    return [f.shape[1] for f in features]

  def forward(self, x):
    features = self.encoder(x)
    encoded = self.decoder(features)
    #output = self.out_conv(encoded)
    cls_out = self.classify(encoded)
    connected_out = self.connected(encoded)
    connected_out = torch.softmax(connected_out, dim=1)
    return connected_out, cls_out


class SuperModelV3(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1, classes=1):
    super(SuperModelV3, self).__init__()
    self.backbone_name = backbone_name
    self.model = smp.FPN(self.backbone_name, classes=last_features)
    self.classify = conv3x3(last_features, classes)
    self.connected = conv3x3(last_features, connected_channels)

  def get_features(self):
    with torch.no_grad():
      features = self.encoder(torch.zeros((1, 3, 64, 64)))
    return [f.shape[1] for f in features]

  def forward(self, x):
    encoded = self.model(x)
    cls_out = self.classify(encoded)
    connected_out = self.connected(encoded)
    connected_out = torch.softmax(connected_out, dim=1)
    return connected_out, cls_out


class SuperModelV4(torch.nn.Module):
  def __init__(self, backbone_name, last_features=32, connected_channels=1):
    super(SuperModelV4, self).__init__()
    self.backbone_name = backbone_name
    self.encoder = timm.create_model(backbone_name, features_only=True)
    self.features_channels = [last_features, *self.get_features()]
    self.decoder = Decoder(features=self.features_channels, return_features=True)
    self.connected = conv3x3(last_features, connected_channels)

  def get_features(self):
    with torch.no_grad():
      features = self.encoder(torch.zeros((1, 3, 64, 64)))
    #[print(f.shape[1]) for f in features]
    return [f.shape[1] for f in features]

  def forward(self, x):
    features = self.encoder(x)
    encoded = self.decoder(features)
    connected_out = self.connected(encoded[-1])
    connected_out = torch.softmax(connected_out, dim=1)
    return connected_out, encoded
