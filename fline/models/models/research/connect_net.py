import torch

from fline.models.blocks.convs import ResidualBlock


class ConnectNet(torch.nn.Module):
    def __init__(
            self,
            features_size,
            connections_classes,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(ConnectNet, self).__init__()
        convs = [ResidualBlock(
            in_channels=features_size,
            out_channels=features_size,
        ) for i in range(n_layers-1)]
        self.layer = torch.nn.Sequential(
            *convs,
            torch.nn.Conv2d(
                in_channels=features_size,
                out_channels=connections_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.activation = activation
        self.norm = torch.nn.BatchNorm2d(features_size)

    def forward(self, vectors_a, vectors_b):
        #print(vectors_a.shape, vectors_b.shape)
        vectors_matrix = torch.matmul(vectors_a, vectors_b.transpose(2, 3))
        #vectors_matrix = self.norm(vectors_matrix)
        bbox = self.layer(vectors_matrix)
        connections = self.activation(bbox)
        return connections


class ConnectNetV2(torch.nn.Module):
    def __init__(
            self,
            features_size,
            connections_classes,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(ConnectNetV2, self).__init__()
        convs = [ResidualBlock(
            in_channels=1,
            out_channels=1,
        ) for i in range(n_layers-1)]
        self.layer = torch.nn.Sequential(
            *convs,
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=connections_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.activation = activation
        self.norm = torch.nn.BatchNorm2d(1)

    def forward(self, vectors_a, vectors_b):
        #print(vectors_a.shape, vectors_b.shape)
        vectors_a = vectors_a.permute(0, 3, 2, 1)
        vectors_b = vectors_b.permute(0, 3, 1, 2)
        vectors_matrix = torch.matmul(vectors_a, vectors_b)
        vectors_matrix = self.norm(vectors_matrix)
        bbox = self.layer(vectors_matrix)
        connections = self.activation(bbox)
        return connections


class ConnectNetV3(torch.nn.Module):
    def __init__(
            self,
            features_size,
            connections_classes,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(ConnectNetV3, self).__init__()
        convs = [ResidualBlock(
            in_channels=features_size,
            out_channels=features_size,
        ) for i in range(n_layers-1)]
        self.layer = torch.nn.Sequential(
            *convs,
            torch.nn.Linear(
                in_features=features_size*2,
                out_features=connections_classes,
            )
        )
        self.activation = activation
        self.norm = torch.nn.BatchNorm2d(features_size)

    def forward(self, vectors_a, vectors_b):
        connections = []
        for i in range(vectors_b.shape[2]):
            cur_a = vectors_b[:, :, i, 0]
            connects_a = []
            for j in range(vectors_a.shape[2]):
                cur_b = vectors_a[:, :, j, 0]
                cur_v = torch.cat([cur_a, cur_b], dim=1)
                connect = self.layer(cur_v)
                connect = self.activation(connect)
                connects_a.append(connect.unsqueeze(dim=2))
            connects_a = torch.cat(connects_a, dim=2)
            connections.append(connects_a.unsqueeze(dim=3))
        connections = torch.cat(connections, dim=3)
        return connections


class ConnectNetV4(torch.nn.Module):
    def __init__(
            self,
            features_size,
            connections_classes,
            activation=torch.nn.Sigmoid(),
            n_layers=1,
    ):
        super(ConnectNetV4, self).__init__()
        features_size = features_size*3
        convs = [ResidualBlock(
            in_channels=features_size,
            out_channels=features_size,
        ) for i in range(n_layers-1)]
        self.layer = torch.nn.Sequential(
            *convs,
            torch.nn.Conv2d(
                in_channels=features_size,
                out_channels=connections_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.activation = activation
        self.norm = torch.nn.BatchNorm2d(features_size)

    def forward(self, vectors_a: torch.Tensor, vectors_b: torch.Tensor):
        #print(vectors_a.shape, vectors_b.shape)
        matrix_vectors = []
        for i in range(vectors_a.shape[2]):
            cur_v = vectors_a[:,:,i:i+1,:].repeat((1, 1, vectors_b.shape[2], 1))
            #print(i, cur_v.shape, vectors_b.shape, vectors_a.shape)
            cur_v = torch.cat([cur_v, vectors_b], dim=1)
            matrix_vectors.append(cur_v)
        matrix_vectors = torch.cat(matrix_vectors, dim=3)
        #print(matrix_vectors.shape)
        vectors_matrix = torch.matmul(vectors_a, vectors_b.transpose(2, 3))
        vectors_matrix = torch.cat([matrix_vectors, vectors_matrix], dim=1)
        #vectors_matrix = self.norm(vectors_matrix)
        bbox = self.layer(vectors_matrix)
        connections = self.activation(bbox)
        return connections