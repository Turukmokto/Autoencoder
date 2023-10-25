import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, B=2):
        super(Decoder, self).__init__()
        self.B = B
        self.convtr1, self.conv1 = self.create_decode_block(512, 256, 256)
        self.convtr2, self.conv2 = self.create_decode_block(256, 128, 128)
        self.convtr3, self.conv3 = self.create_decode_block(128, 64, 64)
        self.convtr4, self.conv4 = self.create_decode_block(64, 32, 32)
        self.convtr5, self.conv5 = self.create_decode_block(32, 16, 3, True)

    def create_decode_block(self, in_channels_count, out_channels_count, batch_norm_count, final=False):
        convtr = nn.ConvTranspose2d(in_channels=in_channels_count, out_channels=out_channels_count, kernel_size=2,
                                    stride=2)
        conv = nn.Sequential(nn.Conv2d(out_channels_count, batch_norm_count, kernel_size=3, padding=1),
                             nn.BatchNorm2d(batch_norm_count))
        if not final:
            conv.append(nn.ReLU(inplace=True))
        return convtr, conv

    def forward(self, x):
        x = x.reshape(x.shape[0], 512, 16, 16)
        x = x.float()

        x = self.convtr1(x)
        x = self.conv1(x) + x

        x = self.convtr2(x)
        x = self.conv2(x) + x

        x = self.convtr3(x)
        x = self.conv3(x) + x

        x = self.convtr4(x)
        x = self.conv4(x) + x

        x = self.convtr5(x)
        x = self.conv5(x)
        return x
