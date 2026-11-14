import torch.nn as nn

from alcv.models.encoder import Encoder
from alcv.models.decoder import Decoder


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.parametrize(mu, logvar)
        output = self.decoder(z)

        return output, z, mu, logvar