import torch
from torch.autograd import Variable
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z=100, h=28, w=28, n=1):

        super(Generator, self).__init__()

        out_size = h * w * n
        self.z = z

        self.generator = nn.Sequential(
            nn.Linear(z, 128),
            nn.ReLU(),
            nn.Linear(128, out_size),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.generator.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.075**2)
                m.bias.data.normal_(0, 0.075**2)
        return

    def forward(self, x):

        x = x.view(x.size(0), -1)
        out = self.generator(x)

        return out


class Discriminator(nn.Module):

    def __init__(self, h=28, w=28, n=1):

        super(Discriminator, self).__init__()

        im_size = h * w * n

        self.discriminator = nn.Sequential(
            nn.Linear(im_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.075 ** 2)
                m.bias.data.normal_(0, 0.075**2)
        return

    def forward(self, x):

        x = x.view(x.size(0), -1)
        out = self.discriminator(x)

        return out


if __name__ == "__main__":

    gen = Generator()
    desc = Discriminator()
