from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import pylab as plt
from utils import *
from copy import deepcopy


#Dataloaders
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

#training routine for a single epoch
def train(model, train_loader, optimizer, criterion, denoiser=False):

    model.train()
    epoch_loss = 0.

    for batch_idx, (data, _) in enumerate(train_loader):

        if denoiser:
            input = add_white_noise(deepcopy(data))
            input = Variable(input)
        else:
            input = Variable(data)

        data = Variable(data)

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, data)
        epoch_loss += loss.data[0]

        loss.backward()

        optimizer.step()

    return epoch_loss

#testing routine
def test(model, test_loader, criterion, num_examples=10, denoiser=False):

    model.eval()
    test_loss = 0.
    examples = []

    for data, _ in test_loader:

        if denoiser:
            data = add_white_noise(data)

        data = Variable(data, volatile=True)

        output = model(data)

        loss = criterion(output, data)
        test_loss += loss.data[0]

        if len(examples) < num_examples:
            for i in range(test_loader.batch_size):
                if len(examples) < num_examples:
                    input = data.data[i, 0].numpy()
                    input = input.reshape(input.shape[0]*input.shape[1])
                    result = output.data[i].numpy()
                    examples.append([input, result])

    return test_loss, examples

#running the training
def run_training(model, train_loader, test_loader, optimizer, criterion, epochs=50, denoiser=False):

    print('runnning training...')
    examples_list = []
    test_losses = []

    for epoch in range(epochs):

        print('\nEpoch:', epoch)

        epoch_loss = train(model, train_loader, optimizer, criterion, denoiser=denoiser)
        print('average training loss:', epoch_loss / len(train_loader.dataset))

        if epoch % 10 == 0 or epoch == epochs - 1:
            test_loss, examples = test(model, test_loader, criterion, denoiser=denoiser)
            print('average loss on test set:', test_loss / len(test_loader.dataset))
            examples_list.append(examples)
            test_losses.append(test_loss)

    return test_losses, np.array(examples_list)

if __name__ == "__main__":

    data, truth = next(iter(train_loader))
    noisy = add_white_noise(data, factor=0.2)
    noisy = noisy.numpy()
    noisy = noisy[0, 0]
    data = data.numpy()
    data = data[0, 0]


    net = Autoencoder()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    losses, ex = run_training(net, train_loader, test_loader, optimizer, criterion, epochs=10, denoiser=True)