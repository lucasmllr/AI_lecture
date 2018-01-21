from VAE import VAE
import torch
import pylab as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.cluster import KMeans

def sample(vae_model, n, title=None, path=None):

    samples = vae_model.sample(n)
    samples = samples.data.numpy()
    samples = samples.reshape(n, vae_model.img_size, vae_model.img_size)

    fig, axes = plt.subplots(1, n)

    for i, ax in enumerate(axes):
        ax.imshow(samples[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0, wspace=0)

    if title:
        plt.suptitle(title)
    if path:
        plt.savefig(path)
    plt.show()

    return

def visualize_trafo(vae_model, n, title=None, path=None):

    #dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=n, shuffle=True)

    data, _ = next(iter(dataloader))
    raw_input = data.numpy()
    input = Variable(data)
    output, _, _ = vae_model(input)
    output = output.data.numpy()
    output = output.reshape(n, vae_model.img_size, vae_model.img_size)

    f, axes = plt.subplots(2, n)

    for i in range(n):

        axes[0, i].imshow(raw_input[i, 0])
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        axes[1, i].imshow(output[i])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    if title:
        plt.suptitle(title)
    if path:
        plt.savefig(path)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

    return

def visualize_cluster_centers(vae_model, n, title=None, path=None):

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=n, shuffle=True)

    input, _ = next(iter(dataloader))
    input = Variable(input)
    codes = vae_model.encode(input)

    codes = codes.view(n, -1)
    codes = codes.data.numpy()

    kmeans = KMeans(n_clusters=10).fit(codes)
    centers = kmeans.cluster_centers_

    centers = Variable(torch.from_numpy(centers))
    center_images = vae_model.decoder(centers)
    center_images = center_images.view(10, vae_model.img_size, vae_model.img_size)
    center_images = center_images.data.numpy()

    fig, axes = plt.subplots(1, 10)

    for i, ax in enumerate(axes):
        ax.imshow(center_images[i])
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title)
    if path:
        plt.savefig(path)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

    return

if __name__ == "__main__":

    vae = VAE(28)
    vae.load_state_dict(torch.load('models/MNIST_VAE_10.pth'))

    #sample(vae, 10, title='samples after 10 training epoch', path='plots/samples after 10 training epoch')
    #visualize_trafo(vae, 10, title='transformations after 10 training epochs', path='plots/trafo_10')
    visualize_cluster_centers(vae, 400, title='cluster centers after 10 epochs', path='plots/clusters_10')