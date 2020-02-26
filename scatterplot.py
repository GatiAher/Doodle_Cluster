import numpy as np

from pca import mean_center_by_row, projection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style



if __name__ == '__main__':

    #####################
    # LOAD NUMPY ARRAYS #
    #####################

    labels = np.load("labels.npy")
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
    print("IMAGE PROJ SHAPE: ", images_proj.shape)

    ####################
    # SCATTERPLOT PLOT #
    ####################

    style.use('ggplot')

    fig = plt.figure()
    # make 3D axis
    ax = fig.add_subplot(111, projection='3d')

    # scatter plots
    ax.scatter(images_proj[:,0], images_proj[:,1], zs=0, zdir='z', s=20, c=None, depthshade=True)

    plt.show()
