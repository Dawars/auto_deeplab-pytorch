import torch.nn as nn

from layers import Cell


class AutoDeeplab(nn.Module):
    def __init__(self, L):
        super(AutoDeeplab, self).__init__()

        self.L = L

        self.cells = []

        for i in range(L):
            # todo dilation on up/downsample
            self.cells.append(Cell(64, 64))

        # self.aspp

    def forward(self, x):

        H = []
        for conv in self.stem_layers:
            x = conv(x)
            H.append(x)

        layers = self.stem_layers + self.cells
        for i in range(len(self.stem_layers), len(layers)):

            x = layers(H[i - 1], H[i - 2])  # fixme not working yet
            H.append(x)

        # x = aspp
        return x


if __name__ == '__main__':
    model = AutoDeeplab(12)
    print(model)
