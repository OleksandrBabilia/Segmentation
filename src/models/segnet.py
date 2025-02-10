from torch import nn 
from models.parts import DoubleConv2d, DoubleUnConv2d, TripleConv2d, TripleUnConv2d

class SegNet(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = DoubleConv2d(3, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = TripleConv2d(128, 256, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = TripleConv2d(256, 512, kernel_size=kernel_size, padding=kernel_size//2)

        self.unconv1 = TripleUnConv2d(512, 256, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv2 = TripleUnConv2d(256, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv3 = DoubleUnConv2d(128, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.unconv4 = DoubleUnConv2d(64, 3, kernel_size=kernel_size, padding=kernel_size//2)

        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.bn(x)
        print(x.shape)

        x, indices1, shape1 = self.conv1(x)
        x, indices2, shape2 = self.conv2(x)
        x, indices3, shape3 = self.conv3(x)
        x, indices4, shape4 = self.conv4(x)

        x = self.unconv1(x, indices4, shape4)
        x = self.unconv2(x, indices3, shape3)
        x = self.unconv3(x, indices2, shape2)
        x = self.unconv4(x, indices1, shape1)

        return x