import torch.nn as nn
import torch
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    """
    https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
    :param kernel_size:
    :param dilation:
    :return:
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, [pad_beg, pad_end, pad_beg, pad_end])
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, batch_norm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = batch_norm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.pointwise(x)
        return x


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, activation=nn.ReLU6, bn=nn.BatchNorm2d):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        :param dilation: dilation rate of 3*3 depthwise conv
        """
        super(Cell, self).__init__()
        self.in_ = in_channels
        self.out_ = out_channels
        self.activation = activation

        self.atr3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation(),
        )
        self.atr5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation(),
        )

        self.sep3x3 = nn.Sequential(SeparableConv2d(in_channels, out_channels, kernel_size=3, batch_norm=bn),
                                    activation())
        self.sep5x5 = nn.Sequential(SeparableConv2d(in_channels, out_channels, kernel_size=5, batch_norm=bn),
                                    activation())

    def forward(self, h_1, h_2):
        """

        :param h_1:
        :param h_2:
        :return:
        """

        top = self.atr5x5(h_2) + self.sep3x3(h_1)
        bottom = self.atr3x3(h_1) + self.sep3x3(h_2)
        middle = self.sep3x3(bottom) + self.sep3x3(h_2)

        top2 = self.sep5x5(top) + self.sep5x5(middle)
        bottom2 = self.atr5x5(top2) + self.sep5x5(bottom)

        concat = torch.cat([top, top2, middle, bottom2, bottom])

        return concat
