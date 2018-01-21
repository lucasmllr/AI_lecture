import torch
import torch.nn as nn
import code
from torch.autograd import Variable


class VAE(nn.Module):

    def __init__(self, img_size):

        super(VAE, self).__init__()

        self.img_size = img_size

        self.encoder = nn.Sequential(
            nn.Linear(img_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, 64)

        self.logvar = nn.Linear(256, 64)

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size**2),
            nn.Tanh()
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        #deriving mu and var for instances x
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        #sampling instances in the surrounding of x
        sample = code.reparameterize(self, mu, logvar)

        #decoding the sample to an image
        rebuilt = self.decoder(sample)

        return rebuilt, mu, logvar

    def encode(self, x):

        x = x.view(x.size(0), -1)

        #deriving mu for instances x
        x = self.encoder(x)
        mu = self.mu(x)

        return mu

    def sample(self, n=10):

        #random samples
        code = Variable(torch.randn(n, 64))
        sample = self.decoder(code)

        return sample

