import torch
from torch import nn
import torch.nn.functional as F

class BlockConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, batch_size=None, kernel_size=3, stride=1,
                 padding=0, dilation=1, bias=False, dropout=False):
        super(BlockConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
        )
        if batch_size:
            self.layer.add_module(name="BatchNorm2d", module=nn.BatchNorm2d(batch_size))
        self.layer.add_module(name="ReLU", module=nn.ReLU())
        if dropout:
            self.layer.add_module(name="DropOut", module=nn.Dropout()) 

    def forward(self, x):
        return self.layer(x)


class DoubleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, batch=True,
                maxpool=True, maxpool_stride=None):
        super(DoubleConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_out, batch_size=channels_out if batch else None, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out if batch else None, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True, stride=maxpool_stride) if maxpool else None
        

    def forward(self, x):
        x = self.layer(x)
        if not self.maxpool:
            return x

        shape = x.shape
        x, indices = self.maxpool(x)
        return x, indices, shape


class TripleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(TripleConv2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
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
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
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
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
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
            nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, stride=stride,
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
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
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
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_out, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
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
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
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
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            DepthwiseSeparableBlockConv2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x


class BlockConvTranspose2d(nn.Module):
    def __init__(self, channels_in, channels_out, batch_size=None, kernel_size=3, stride=1,
                 padding=0, dilation=1, bias=False):
        super(BlockConvTranspose2d, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
        )
        if batch_size:
            self.layer.add_module(nn.BatchNorm2d(batch_size))
        self.layer.add_module(nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class DoubleUnConvTranspose2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(DoubleUnConvTranspose2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConvTranspose2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConvTranspose2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x
        

class TripleUnConvTranspose2d(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(TripleUnConvTranspose2d, self).__init__()
        self.layer = nn.Sequential(
            BlockConvTranspose2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConvTranspose2d(channels_in, channels_in, batch_size=channels_in, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias),
            BlockConvTranspose2d(channels_in, channels_out, batch_size=channels_out, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias)
        )

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size=output_size)
        x = self.layer(x)
        return x


class UnetUpDoubleConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels_in, channels_in//2, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(channels_in, channels_out, kernel_size=kernel_size, padding=padding, batch=True, maxpool=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)