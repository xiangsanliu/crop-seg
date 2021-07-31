# from .mix_transformer import mit_b0, mit_b1, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
import model.segformer.mix_transformer as encoders
from .segformer_decoder import SegFormerHead
import torch.nn as nn


class Segformer(nn.Module):
    def __init__(self, encode_config, decoder_config):
        super().__init__()
        self.encoder = getattr(encoders, encode_config['type'])()
        self.encoder.load_pretained(encode_config['pretrained'])
        self.decoder = SegFormerHead(**decoder_config)
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x
