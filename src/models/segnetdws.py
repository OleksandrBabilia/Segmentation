from torch import nn 
from models.parts import DepthwiseSeparableDoubleConv2d, DepthwiseSeparableDoubleUnConv2d, DepthwiseSeparableTripleConv2d, DepthwiseSeparableTripleUnConv2d

class DWSSegNet(nn.Module):
    def __init__(self, kernel_size):
        super(DWSSegNet, self).__init__()

        self.conv1 = DepthwiseSeparableDoubleConv2d(3, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = DepthwiseSeparableDoubleConv2d(64, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = DepthwiseSeparableTripleConv2d(128, 256, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = DepthwiseSeparableTripleConv2d(256, 512, kernel_size=kernel_size, padding=kernel_size//2)

        self.unconv1 = DepthwiseSeparableTripleUnConv2d(512, 256, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv2 = DepthwiseSeparableTripleUnConv2d(256, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv3 = DepthwiseSeparableDoubleUnConv2d(128, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv4 = DepthwiseSeparableDoubleUnConv2d(64, 3, kernel_size=kernel_size, padding=kernel_size//2)

        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.bn(x)

        x, indices1, shape1 = self.conv1(x)
        x, indices2, shape2 = self.conv2(x)
        x, indices3, shape3 = self.conv3(x)
        x, indices4, shape4 = self.conv4(x)

        x = self.unconv1(x, indices4, shape4)
        x = self.unconv2(x, indices3, shape3)
        x = self.unconv3(x, indices2, shape2)
        x = self.unconv4(x, indices1, shape1)

        return x
        