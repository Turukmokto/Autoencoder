import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Autoencoder(nn.Module):
    def __init__(self, B=2):
        super(Autoencoder, self).__init__()
        self.B = B
        self.encoder = Encoder(self.B)
        self.decoder = Decoder(self.B)

    def forward(self, x):
        x = self.encoder(x)
        x = x + (1 / 2 ** self.B) * (torch.rand_like(x) * 0.5 - 0.5)
        x = self.decoder(x.float())
        x = torch.sigmoid(x)
        return x
