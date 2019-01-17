import torch
import torch.nn as nn
import layers

class AutoDeeplab(nn.Module):
    def __init__(self, in_channels, out_channels, layout, cell, activation=nn.ReLU6, upsample_at_end=True):
        """
        A general implementation of the network architecture presented in the Auto Deeplab paper
        :param layout: A list of integers representing the y coordinate of a cell in the diagram used in the paper (zero-indexed)
        :param cell: The cell class to use.
        """
        super(AutoDeeplab, self).__init__()

        cells = []
        channels = in_channels
        for i, depth in enumerate(layout):
            layer = i + 1
            size = 2 ** depth
            if i != 0:
                prev_depth = layout[i - 1]
                assert abs(depth - prev_depth) <= 1
                if prev_depth < depth:
                    # Downsampling
                    cells.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
                    channels = channels * 2
                elif prev_depth > depth:
                    # Upsampling
                    cells.append(nn.Upsample(scale_factor=2, mode="bilinear"))
                    cells.append(nn.Conv2d(channels, channels // 2, 1))
                    channels = channels // 2

            # channels = F * B * size
            # todo dilation?
            cells.append(
                cell(channels, channels)
            )
            cells.append(activation())

        cells.append(layers.ASPP(channels, 256, (6, 12, 18), (6, 12, 18)))
        channels = 256

        # Reduce channels to the desired value
        cells.append(nn.Conv2d(channels, out_channels, 3, padding=1))

        if upsample_at_end:
            cells.append(nn.Upsample(scale_factor=2 ** layout[-1], mode="bilinear"))

        self.network = nn.Sequential(*cells).cuda()

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    layout = [0, 1, 2, 2, 2, 2, 3, 4, 3, 4, 4, 5, 5, 4, 3]
    model = AutoDeeplab(3, 3, layout, layers.Cell)
    print(model)
