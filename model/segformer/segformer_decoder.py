import torch
import torch.nn as nn
from model.utils import resize
# from mmcv.cnn import ConvModule


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


class SegFormerHead(nn.Module):
    def __init__(
        self,
        feature_strides,
        in_channels,
        embed_dim,
        in_index,
        num_classes,
        dropout_ratio,
    ):
        super(SegFormerHead, self).__init__()
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

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = (
            self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        )
        _c4 = resize(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = (
            self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        )
        _c3 = resize(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = (
            self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        )
        _c2 = resize(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = (
            self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        )
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

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



