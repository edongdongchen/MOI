import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, residual=True, circular_padding=False, cat=True):
        super(UNet, self).__init__()
        """compact unet (4 levels)"""
        self.name = 'unet'

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        def conv_block(ch_in, ch_out, circular_padding=False):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True,
                          padding_mode='circular' if circular_padding else 'zeros'),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def up_conv(ch_in, ch_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64, circular_padding=circular_padding)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        cat_dim = 1
        input = x

        x1 = self.Conv1(input) # 1->64

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)# 64->128

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)# 128->256

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)# 256->512

        d4 = self.Up4(x4) # 512->256
        if self.cat:
            d4 = torch.cat((x3, d4), dim=cat_dim)
            d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)# 256->128
        if self.cat:
            d3 = torch.cat((x2, d3), dim=cat_dim)
            d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)# 128->64
        if self.cat:
            d2 = torch.cat((x1, d2), dim=cat_dim)
            d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = d1+x if self.residual and self.in_channels==self.out_channels else d1
        return out