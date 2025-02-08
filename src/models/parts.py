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


class DepthwiseSeparableBlockConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, batch_size, kernel_size=3, stride=1,
                 padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableBlockConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, stride=1,
                      padding=padding, dilation=dilation, bias=bias, groups=channels_in), # depthwise
            nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=bias), # pointwise 
            nn.BatchNorm2d(batch_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DepthwiseSeparableDoubleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableDoubleConv2d, self).__init__()
        self.layer = nn.Sequential(
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.layer(x)
        shape = x.shape
        x, indices = self.maxpool(x)
        return x, indices, shape
    
    
class DepthwiseSeparableTripleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableTripleConv2d, self).__init__()
        self.layer = nn.Sequential(
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.layer(x)
        shape = x.shape
        x, indices = self.maxpool(x)
        return x, indices, shape


class DepthwiseSeparableDoubleUnConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableDoubleUnConv2d, self).__init__()
        self.layer = nn.Sequential(
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x


class DepthwiseSeparableTripleUnConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableTripleUnConv2d, self).__init__()
        self.layer = nn.Sequential(
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=1,
                        padding=kernel_size//2, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x