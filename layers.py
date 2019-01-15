import torch.nn as nn
import torch


class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, s=1, dilation=1):
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
        self.t = t
        self.s = s
        self.dilation = dilation

        self.atr3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), dilation=dilation)
        self.atr5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=dilation)
        # todo sep conv
        self.sep3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=dilation),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        self.sep5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), dilation=dilation)

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
