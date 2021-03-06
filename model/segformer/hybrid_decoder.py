import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings

# from mmcv.cnn import ConvModule

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class HybridHeader(nn.Module):
    def __init__(
        self,
        feature_strides,
        in_channels,
        embed_dim,
        in_index,
        num_classes,
        dropout_ratio,
    ):
        super(HybridHeader, self).__init__()
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_index = in_index
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

        embedding_dim = embed_dim

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # self.linear_fuse = ConvModule(
        #     in_channels=embedding_dim*4,
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim, eps=1e-5),
            nn.ReLU(inplace=True),
        )

        # self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.up1 = UpConv(embedding_dim + 256, 64)
        self.se1 = SELayer(embedding_dim + 256)
        self.up0 = UpConv(64 * 2, self.num_classes)
        self.se0 = SELayer(64 * 2)

    def forward(self, inputs, res):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        stage1, stage2 = res
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = resize(_c4, size=128, mode="bilinear", align_corners=False)

        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = resize(_c3, size=128, mode="bilinear", align_corners=False)

        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = resize(_c2, size=128, mode="bilinear", align_corners=False)

        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )
        _c1 = resize(_c1, size=128, mode="bilinear", align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        _x = torch.cat([x, stage2], dim=1)
        _x = self.se1(_x)
        x = self.up1(_x)

        _x = torch.cat([x, stage1], dim=1)
        _x = self.se0(_x)
        x = self.up0(_x)

        return x

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        inputs = [inputs[i] for i in self.in_index]

        return inputs


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
