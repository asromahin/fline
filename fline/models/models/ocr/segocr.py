import segmentation_models_pytorch as smp
import torch
import typing as tp


class SegOcrModel(torch.nn.Module):
    def __init__(
            self,
            backbone: str,
            input_size: tp.Tuple[int, int],
            rnn_size: int,
            letters_count: int,
    ):
        super(SegOcrModel, self).__init__()
        self.seg_model = smp.Unet(
            backbone,
            in_channels=3,
            classes=letters_count,
            #encoder_weights=None,
            decoder_attention_type='scse',
        )
        self.conv = torch.nn.Conv2d(
            in_channels=letters_count,
            out_channels=letters_count,
            kernel_size=(input_size[0], input_size[1]//rnn_size),
            stride=(input_size[0], input_size[1]//rnn_size),
        )
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.seg_model(x)
        x = self.conv(x)
        x = x.squeeze(dim=2)
        x = x.permute(2, 0, 1)
        x = self.softmax(x)
        return x
