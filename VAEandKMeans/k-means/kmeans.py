import numpy as np
import pylab as plt
from scipy.ndimage import imread
from sklearn.cluster import KMeans

def compress_RGB(image, k):
    '''function to compress the given image

    Args:
        image (ndarray): RGB image in numpy array format of shape (H, W, 3)
        k (integer): number of clusters to fit
    Returns:
        compressed image of same dimension as input'''

    flat = image.reshape(image.shape[0]*image.shape[1], 3)

    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(flat)
    centers = kmeans.cluster_centers_

    compressed = []

    for i in range(flat.shape[0]):
        cluster = labels[i]
        compressed.append(centers[cluster])

    compressed = np.array(compressed)

    return compressed.reshape(image.shape[0], image.shape[1], 3)

if __name__ == "__main__":

    pic = imread('grass2.jpg')
    frac = pic[0:10, 0:10]

    compressed = compress_RGB(pic, 10)

    plt.imshow(compressed)
    plt.show()