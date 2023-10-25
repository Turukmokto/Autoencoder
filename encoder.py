import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, B=2):
        super(Encoder, self).__init__()
        self.B = B
        self.enc = models.resnet18(pretrained=True)
        self.enc = nn.Sequential(*list(self.enc.children())[:-2])
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.enc(x)
        x = self.sigm(x)
        x = x.flatten(1)
        return x
