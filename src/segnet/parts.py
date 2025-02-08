from torch import nn


class BlockConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, batch_size, kernel_size=3, stride=1,
                 padding=0, dilation=1, bias=False):
        super(BlockConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=1,
                      padding=kernel_size//2, dilation=dilation, bias=bias),
            nn.BatchNorm2d(batch_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DoubleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DoubleConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.layer(x)
        shape = x.shape
        x, indices = self.maxpool(x)
        return x, indices, shape


class TripleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(TripleConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.layer(x)
        shape = x.shape
        x, indices = self.maxpool(x)
        return x, indices, shape
        

class DoubleUnConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DoubleUnConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x
        
class TripleUnConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(TripleUnConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x


class SegNet(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = DoubleConv2d(3, 64, kernel_size=kernel_size)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=kernel_size)
        self.conv3 = TripleConv2d(128, 256, kernel_size=kernel_size)
        self.conv4 = TripleConv2d(256, 512, kernel_size=kernel_size)

        self.unconv1 = TripleUnConv2d(512, 256, kernel_size=kernel_size)
        self.unconv2 = TripleUnConv2d(256, 128, kernel_size=kernel_size)
        self.unconv3 = DoubleUnConv2d(128, 64, kernel_size=kernel_size)
        self.unconv4 = DoubleUnConv2d(64, 3, kernel_size=kernel_size)

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