

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x 
class DownSimplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return F.max_pool2d(self.conv(x),2,2)

class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        diff = torch.tensor([x2.size(2) - x1.size(2)])
        padding_x = diff // 2
        padding_y = diff - diff // 2
        x1 = F.pad(x1, [padding_x, padding_y, padding_x, padding_y])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels):
        super(UNet, self).__init__()

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownSimplingBlock(64, 128)
        self.down2 = DownSimplingBlock(128, 256)
        self.down3 = DownSimplingBlock(256, 512)
        self.down4 = DownSimplingBlock(512, 1024)
        self.up1 = UpsamplingBlock(1024, 512)
        self.up2 = UpsamplingBlock(512, 256)
        self.up3 = UpsamplingBlock(256, 128)
        self.up4 = UpsamplingBlock(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, out_channels=2)
    input_tensor = torch.randn(1, 3, 696, 696)
    output_tensor = net(input_tensor)
    print(output_tensor.shape)