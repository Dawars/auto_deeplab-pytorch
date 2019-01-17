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
        self.upsample_at_end = upsample_at_end
        self.cells = []

        self.initial_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation()
        ).cuda()

        self.cells.append(nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                activation()
        ))

        self.cells.append(nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation()
        ))


        #self.stem = nn.Sequential(
        #    nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
        #    nn.Conv2d(64, 64, 3, padding=1),
        #    nn.Conv2d(64, 128, 3, stride=2, padding=1),
        #).cuda()

        channels = 128
        assert layout[0] == 2
        for i, depth in enumerate(layout):
            layer = [cell(channels, channels)]
            # todo dilation?

            if i != len(layout) - 1:
                next_depth = layout[i + 1]
                assert abs(depth - next_depth) <= 1
                if next_depth > depth:
                    # Downsampling
                    layer.append(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1))
                    channels = channels * 2
                elif next_depth < depth:
                    # Upsampling
                    layer.append(nn.Upsample(scale_factor=2, mode="bilinear"))
                    layer.append(nn.Conv2d(channels, channels // 2, 1))
                    channels = channels // 2

            self.cells.append(nn.Sequential(*layer).cuda())

        # Pool, then reduce channels to the desired value
        self.pool = nn.Sequential(
            layers.ASPP(channels, 256, (6, 12, 18), (6, 12, 18)),
            nn.Conv2d(256, out_channels, 3, padding=1)
        ).cuda()

        self.upsampler = nn.Upsample(scale_factor=2 ** layout[-1], mode="bilinear")


    def forward(self, x):
        x = self.initial_stem(x)

        # Run stem layers
        prev_hs = [self.cells[0](x)]
        prev_hs.append(self.cells[1](prev_hs[0]))

        for i, layer in enumerate(self.cells, 2):
            curr = layer(prev_hs[-1], prev_hs[-2])
            prev_hs[-2] = prev_hs[-1]
            prev_hs[-1] = curr

        x = self.pool(prev_hs[-1])
        if self.upsample_at_end:
            x = self.upsampler(x)

        return x


if __name__ == '__main__':
    layout = [2, 2, 2, 2, 3, 4, 3, 4, 4, 5, 5, 4, 3]
    model = AutoDeeplab(3, 3, layout, layers.Cell)
    print(model)

