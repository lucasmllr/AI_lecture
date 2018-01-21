import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.autograd import Variable
import code
import os
from VAE import VAE

#Dataloader
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

#training function
def train(model, data_loader, optimizer, epochs, path=None):

    if path and os.path.exists(path):
        raise IOError('path to save model already exists. provide another.')

    model.train()

    for epoch in range(epochs):
        print('started epoch', epoch)
        for (data, _) in data_loader:

            input = Variable(data)

            optimizer.zero_grad()

            rebuilt, mu, logvar = model(input)

            loss = code.loss_function(rebuilt, input, mu, logvar, batch_size=128, img_size=28, nc=1)

            loss.backward()

            optimizer.step()

    if path:
        torch.save(model.state_dict(), path)

    return

if __name__ == "__main__":

    vae = VAE(img_size=28)
    optimizer = Adam(vae.parameters(), lr=0.001)

    train(vae, data_loader, optimizer, epochs=10, path='models/MNIST_VAE_10.pth')


