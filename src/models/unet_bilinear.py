from torch import nn
from models.parts import DoubleConv2d, UnetUpDoubleConv2d, TripleConv2d, TripleUnConvTranspose2d, BlockConv2d

class UNetBilinear(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = DoubleConv2d(3, 64, kernel_size=kernel_size, padding=kernel_size//2, batch=False, maxpool_stride=2)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=kernel_size, padding=kernel_size//2, batch=False, maxpool_stride=2)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=kernel_size, padding=kernel_size//2, batch=False, maxpool_stride=2)
        self.conv4 = DoubleConv2d(256, 512, kernel_size=kernel_size, padding=kernel_size//2, batch=False, maxpool_stride=2)

        self.bottle_neck = DoubleConv2d(512, 512, kernel_size=3, batch=False, maxpool=False)

        self.upconv1 = UnetUpDoubleConv2d(1024, 256, kernel_size=kernel_size, padding=kernel_size//2, bilinear=True)
        self.upconv2 = UnetUpDoubleConv2d(512, 128, kernel_size=kernel_size, padding=kernel_size//2, bilinear=True)
        self.upconv3 = UnetUpDoubleConv2d(256, 64, kernel_size=kernel_size, padding=kernel_size//2, bilinear=True)
        self.upconv4 = UnetUpDoubleConv2d(128, 64, kernel_size=kernel_size, padding=kernel_size//2, bilinear=True)
        self.out = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x1, _, _ = self.conv1(x)
        x2 , _, _= self.conv2(x1)
        x3, _, _ = self.conv3(x2)
        x4, _, _ = self.conv4(x3)

        x = self.bottle_neck(x4)

        x = self.upconv1(x, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)
        x = self.out(x)

        return x
