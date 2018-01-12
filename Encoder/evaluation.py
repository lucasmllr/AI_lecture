from tsne import *
from training import *
import pylab as plt

def plot_ex(examples, epoch):
    num_examples = examples.shape[0]

    f, ax = plt.subplots(2, num_examples)

    f.suptitle('transformations of epoch {}'.format(epoch))

    for i in range(num_examples):
        input = examples[i, 0].reshape(28, 28)
        output = examples[i, 1].reshape(28, 28)
        ax[0, i].imshow(input)
        ax[1, i].imshow(output)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

    return

def visualize_codes(net, dataloader=test_loader, batches=4):

    codes = []
    truths = []

    for _ in range(batches):

        to_encode, truth = next(iter(dataloader))
        truths.append(truth.numpy())
        to_encode = Variable(to_encode)
        encoded = net.encode(to_encode)
        codes.append(encoded.data.numpy())

    X = np.concatenate(codes, axis=0)
    GT = np.concatenate(truths, axis=0)
    Y = tsne(X, no_dims=2, initial_dims=8)

    tops = Y[np.where(GT==0)]
    trousers = Y[np.where(GT == 1)]
    pullovers = Y[np.where(GT == 2)]
    dresses = Y[np.where(GT == 3)]
    coats = Y[np.where(GT == 4)]
    sandals = Y[np.where(GT == 5)]
    shirts = Y[np.where(GT == 6)]
    sneakers = Y[np.where(GT == 7)]
    bags = Y[np.where(GT == 8)]
    boots = Y[np.where(GT == 9)]

    plt.scatter(tops[:, 0], tops[:, 1], label='tops')
    plt.scatter(trousers[:, 0], trousers[:, 1], label='trousers')
    plt.scatter(pullovers[:, 0], pullovers[:, 1], label='pullovers')
    plt.scatter(dresses[:, 0], dresses[:, 1], label='dresses')
    plt.scatter(coats[:, 0], coats[:, 1], label='coats')
    plt.scatter(sandals[:, 0], sandals[:, 1], label='sandals')
    plt.scatter(shirts[:, 0], shirts[:, 1], label='shirts')
    plt.scatter(sneakers[:, 0], sneakers[:, 1], label='sneakers')
    plt.scatter(bags[:, 0], bags[:, 1], label='bags')
    plt.scatter(boots[:, 0], boots[:, 1], label='boots')

    plt.title('visualization of codes')
    plt.legend()
    plt.show()

    return X, Y, GT

if __name__ == "__main__":

    net2 = Autoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)

    losses, ex = run_training(net2, train_loader, test_loader, optimizer, criterion, epochs=10, denoiser=True)

    plot_ex(ex[-1], 10)
    codes, reduced, truth = visualize_codes(net2, test_loader)
    #print(codes.shape)
    #print(truth.shape)
    #print(reduced.shape)