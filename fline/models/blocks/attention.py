import torch

from fline.models.models.research.connected_components import ConnectedComponents


class SelfAttention(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        #print(proj_query.shape, proj_key.shape, x.shape)
        energy = torch.matmul(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.matmul(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class SelfAttentionPatch(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, downsample=4):
        super(SelfAttentionPatch, self).__init__()
        self.chanel_in = in_dim
        self.downsample = downsample

        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)  #

        self.pool = torch.nn.MaxPool2d(kernel_size=(self.downsample, self.downsample), stride=(self.downsample, self.downsample))
        self.up = torch.nn.UpsamplingNearest2d(scale_factor=self.downsample)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.pool(self.query_conv(x)).view(m_batchsize, -1, width * height//(self.downsample**2)).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.pool(self.key_conv(x)).view(m_batchsize, -1, width * height//(self.downsample**2))  # B X C x (*W*H)
        #print(proj_query.shape, proj_key.shape, x.shape)
        energy = torch.matmul(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.pool(self.value_conv(x)).view(m_batchsize, -1, width * height//(self.downsample**2))  # B X C X N

        out = torch.matmul(proj_value, attention.permute(0, 2, 1))
        #print(out.shape)
        out = self.up(out.view(m_batchsize, C, width//self.downsample, height//self.downsample))

        out = self.gamma * out + x
        return out, attention


class SelfAttentionPatchV2(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, downsample=4):
        super(SelfAttentionPatchV2, self).__init__()
        self.attn = SelfAttention(in_dim=in_dim)
        self.downsample = downsample

        self.pool = torch.nn.MaxPool2d(
            kernel_size=(self.downsample, self.downsample),
            stride=(self.downsample, self.downsample),
        )
        self.up = torch.nn.UpsamplingNearest2d(scale_factor=self.downsample)

    def forward(self, x):
        x_prepare = self.pool(x)
        out, out_attn = self.attn(x_prepare)
        out = self.up(out)
        out_attn = self.up(out_attn)
        return out, out_attn


class ConnCompAttention(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, downsample=4):
        super(ConnCompAttention, self).__init__()
        self.conncomp = ConnectedComponents()
        self.downsample = downsample
        self.attn_conv = torch.nn.Conv2d(
            in_channels=in_dim, out_channels=3, kernel_size=3, stride=1, padding=1,
        )
        self.softmax = torch.nn.Softmax2d()
        self.limit = 20
        self.norm = torch.nn.BatchNorm2d(in_dim)

    def forward(self, x: torch.Tensor):
        out_attn = self.softmax(self.attn_conv(x))
        out_attn_argmax = out_attn.argmax(dim=1)
        out1 = self.conncomp(out_attn_argmax == 1)
        out2 = self.conncomp(out_attn_argmax == 2)
        for b in range(x.shape[0]):
            vectors1 = []
            out_uniq1 = torch.unique(out1[b])
            out_uniq2 = torch.unique(out2[b])
            if len(out_uniq1) == 1 and len(out_uniq2) == 1:
                continue
            for attn_area_val in out_uniq1:
                if attn_area_val != 0:
                    attn_area = (out1[b] == attn_area_val)
                    data_area = x[b][:, attn_area].mean(dim=1, keepdim=True)   #[0] # (C, 1)
                    vectors1.append(data_area)
            vectors1 = torch.cat(vectors1, dim=1)   # (C, N)

            vectors2 = []

            for attn_area_val in out_uniq2:
                if attn_area_val != 0:
                    attn_area = (out2[b] == attn_area_val)
                    data_area = x[b][:, attn_area].mean(dim=1, keepdim=True)  # [0] # (C, 1)
                    vectors2.append(data_area)
            vectors2 = torch.cat(vectors2, dim=1)  # (C, N)

            mm = torch.matmul(vectors1.transpose(0, 1), vectors2)

            for i, attn_area_val1 in enumerate(out_uniq1[:self.limit]):
                if attn_area_val1 != 0:
                    attn_area1 = (out1[b] == attn_area_val1)
                    for j, attn_area_val2 in enumerate(out_uniq2[:self.limit]):
                        if attn_area_val2 != 0:
                            attn_area2 = (out2[b] == attn_area_val2)
                            cur_vec = mm[i - 1, j - 1]
                            #print(i, len(out_uniq1), j, len(out_uniq2))
                            if attn_area2.sum() > 0:
                                x[b][:, attn_area2] += x[b][:, attn_area2]*cur_vec
                            if attn_area1.sum() > 0:
                                x[b][:, attn_area1] += x[b][:, attn_area1]*cur_vec
        x = self.norm(x)
        return x, out_attn
