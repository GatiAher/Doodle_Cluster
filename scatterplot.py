import numpy as np

from pca import mean_center_by_row, projection
from k_means import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style



def convert_labels(labels):
    """ 
    Converts a array of string labels into an array of interger labels
    """
    lookup = np.unique(labels)
    conversion = [lookup[s] for s in labels]
    return np.array(conversion)

if __name__ == '__main__':

    #####################
    # LOAD NUMPY ARRAYS #
    #####################

    labels = np.load("labels.npy")
    new_labels = convert_labels(labels)
    i = 3000
    print(labels[i])
    print(new_labels)
    print(labels[new_labels[i]])
    input()
    print("LABELS SHAPE: ", labels.shape)

    images = np.load("images.npy")
    print("IMAGES SHAPE: ", images.shape)
    # vectorize
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    print("IMAGES SHAPE: ", images.shape)

    vecs = np.load("images-eig-vec.npy")
    print("EIG VECS SHAPE: ", vecs.shape)

    ##############
    # PROJECTION #
    ##############

    # mean center each image
    images_m = mean_center_by_row(images)

    # projection
    k = 3
    images_proj = projection(images_m, vecs, k)
    images_proj = images_proj[:10000, :]
    print("IMAGE PROJ SHAPE: ", images_proj.shape)

    ####################
    # SCATTERPLOT PLOT #
    ####################

    style.use('ggplot')

    fig = plt.figure()
    # make 3D axis
    ax = fig.add_subplot(111, projection='3d')

    # scatter plots
    ax.scatter(images_proj[:,0], images_proj[:,1], images_proj[:,2], zdir='z',
               s=10, c="blue", depthshade=True)

    # cluster and plot
    n_clusters = 345
    k_means = KMeans(n_clusters)
    error = k_means.fit(images_proj)
    labels = k_means.classify_centroids(images_proj, labels[:10000])
    centroids = k_means.centroids
    print(centroids)
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], zdir='z', s=500,
               c="red", depthshade=True)

    plt.show()
