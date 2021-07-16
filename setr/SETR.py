from setr.encoder import Encoder
from setr.decoder import Decoder
import torch.nn as nn

class SETRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(in_channel=config.embed_dim,
                               out_channel=config.out_channel, features=config.decode_features)

    def forward(self, x):
        _, x = self.encoder(x)
        x = self.decoder(x)
        return x
