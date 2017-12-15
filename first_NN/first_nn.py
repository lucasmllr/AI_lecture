import numpy as np
import pylab as plt
from utils import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

#defining models
model1 = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
)

model2 = torch.nn.Sequential(
            torch.nn.Linear(2, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
)

#training
def train(model, train_data, test_data, epochs, learning_rate=0.001):

    loss_fun = torch.nn.MSELoss()
    lr = learning_rate

    for i in range(epochs):

        if i == round(epochs / 2):
            lr = lr / 10

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        epoch_loss = 0

        for j in range(len(train_data)):

            #conversion of training instance and ground truth
            x, y = train_data[j]
            x = Variable(torch.from_numpy(x))
            x = x.float()
            y = Variable(torch.from_numpy(np.array([y])))
            y = y.float()

            #prediction
            y_pred = model(x)

            #computing loss
            loss = loss_fun(y_pred, y)
            epoch_loss += loss

            #zero gradient
            model.zero_grad()

            #propagating loss
            loss.backward()

            #updating weights
            optimizer.step()

        if i % 10 == 0:
            #print('\naverage loss in epoche', i, ':', (epoch_loss.data[0] / len(train_data)))
            test_accuracy = evaluate(model, test_data)
            print('test accuracy in epoche', i, ':', test_accuracy)

    return model

def evaluate(model, test_data):

    total_correct = 0

    for i in range(len(test_data)):

        #data preparation
        x, y = test_data[i]
        x = Variable(torch.from_numpy(x))
        x = x.float()

        #prediction
        y_pred = model(x)
        estimate = y_pred.data.numpy()
        estimate = np.clip(estimate, 0, 3)
        estimate = np.around(estimate)[0]

        #counting correct predictions
        #as a final estimation y_pred values are simply rounded
        if estimate == y:
            total_correct += 1

    accuracy = total_correct / len(test_data)

    return accuracy

def grid_evaluation(model, data=None, round=True):

    x_grid = create_grid_coords()
    y_grid = np.empty(2500)

    for i in range(len(x_grid)):

        x = Variable(torch.from_numpy(x_grid[i]))
        x = x.float()
        y_grid[i] = model(x)

    grid = y_grid.reshape((50,50))
    if round:
        grid = np.around(grid)

    plt.imshow(grid)

    if data != None:
        n = len(data)
        for i in range(n):
            x, y = data[i]
            x  = (x + 1)*25
            if y == 0:
                plt.plot(x[0], x[1], 'ro')
            elif y == 1:
                plt.plot(x[0], x[1], 'go')
            else:
                plt.plot(x[0], x[1], 'bo')

    plt.show()

    return

if __name__ == '__main__':

    #loading the data
    triple = ToyDataset('triple_junction_data_training.txt')
    triple_test = ToyDataset('triple_junction_data_test.txt')

    train(model2, triple, triple_test, 100)

    #grid evaluations of trained net
    grid_evaluation(model2, triple, round=False)


