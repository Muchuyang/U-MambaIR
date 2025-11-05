##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####

import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)

        self.dec4 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec4 = F.interpolate(dec4, size=enc3.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.dec3(dec4)
        dec3 = F.interpolate(dec3, size=enc2.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.dec2(dec3)
        dec2 = F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)


        out = self.relu(self.pixel_shuffle1(self.upconv1(dec2)))
        out = self.relu(self.pixel_shuffle2(self.upconv2(out)))
        out = self.relu(self.HRconv(out))
        out = self.dec1(out)

        return out