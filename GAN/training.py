import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import model
from torchvision import datasets, models, transforms, utils
from torch.utils import data
import os
import evaluation
import numpy as np

#hyperparameters
batch_size = 64
lr = 0.001

D_noise = 100
D_side = 28
D_img = D_side**2
D_hidden = 128

#data
data_set = datasets.MNIST('data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = data.DataLoader(data_set, batch_size, shuffle=True, drop_last=True)

#training
def train_GAN(G, D, data_loader, epochs, name, every_epoch=None):

    if name and os.path.isdir('evaluation/' + name):
        raise IOError('path already exists')
    elif name:
        os.mkdir('evaluation/' + name)

    G_opt = Adam(G.parameters(), lr=lr)
    D_opt = Adam(D.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print('epoch:', epoch)
        for imgs, _ in data_loader:

            #train discriminator
            D.train()
            D_opt.zero_grad()
            #real images
            imgs = Variable(imgs)
            real_truth = Variable(torch.ones(batch_size))
            real_guess = D(imgs).squeeze()
            real_loss = criterion(real_guess, real_truth)

            #fake images
            noise = Variable(torch.randn(batch_size, D_noise))
            fake_truth = Variable(torch.zeros(batch_size))
            fake_guess = D(G(noise)).squeeze()
            fake_loss = criterion(fake_guess, fake_truth)

            #total loss
            D_loss = real_loss + fake_loss
            #backpropagation
            D_loss.backward()
            D_opt.step()
            D.eval()

            #train generator
            G.train()
            G_opt.zero_grad()
            noise = Variable(torch.randn(batch_size, D_noise))
            fake_guess = D(G(noise))
            loss_G = criterion(fake_guess, real_truth)
            loss_G.backward()
            G_opt.step()
            G.eval()

        #evaluation
        if every_epoch:
            if epoch % every_epoch == 0 or epoch == epochs - 1:
                noise = Variable(torch.randn(batch_size, D_noise))
                fake_imgs = G(noise)
                fake_imgs = fake_imgs.view(batch_size, 1, D_side, D_side)
                out = utils.make_grid(fake_imgs.data)
                evaluation.imshow(out, path='evaluation/{}/{}_{}_epochs.pdf'.format(name, name, epoch),
                                  title='{} after {} epochs'.format(name, epoch))

        #saving the model
        torch.save(G.state_dict(), 'evaluation/{}/generator.pth'.format(name))
        torch.save(D.state_dict(), 'evaluation/{}/discriminator.pth'.format(name))

    return


if __name__ == "__main__":

    G = model.Generator()
    G.init_weights()

    D = model.Discriminator()
    D.init_weights()

    train_GAN(G, D, data_loader, epochs=100, name='MNIST_GAN_100', every_epoch=10)