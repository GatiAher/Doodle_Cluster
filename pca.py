import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

#############
# FUNCTIONS #
#############

def mean_center_by_row(images):
    """
    subtract mean of row from each value in row of matrix
    return: ndarray
    """
    # mean of image (mean of rows)
    mean_images = np.mean(images, axis=1)
    print("MEAN IMAGES SHAPE: ", mean_images.shape)

    # make mean of images a column vector
    mean_images = mean_images[:, np.newaxis]
    print("MEAN IMAGES SHAPE: ", mean_images.shape)

    # subtract mean of image from each image (row)
    # uses numpy broadcast to efficiently fill in missing mean columns
    images_m = images - mean_images
    print("IMAGES_M SHAPE: ", images_m.shape)

    # check that subtracting mean was sucessful (images_m.mean(axis=1) = 0)
    # print(images_m.mean(axis=1))

    return images_m



def projection(mean_centered_images, eigenvectors, k):
    """returns projection of shape (num_images, k)
    """
    k_vecs = eigenvectors[:, :k]
    return mean_centered_images @ k_vecs



if __name__ == '__main__':

    ############
    # GET DATA #
    ############

    # load numpy arrays
    labels = np.load("labels.npy")
    print("LABELS SHAPE: ", labels.shape)
    images = np.load("images.npy")
    print("IMAGES SHAPE: ", images.shape)

    # vectorize
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    print("IMAGES SHAPE: ", images.shape)

    ##################
    # NORMALIZE DATA #
    ##################

    # QUESTION: can we mean normalize by row?

    print()
    images_m = mean_center_by_row(images)
    print("IMAGES_M SHAPE: ", images_m.shape)

    # correlation matrix
    # QUESTION: is N number of cols if mean normalize by row?
    # 1/(N - 1) * A' * A
    ata = np.dot(images_m.transpose(), images_m)
    correlation_matrix = (1/(images_m.shape[1] - 1)) * ata
    print("CORRELATION_MATRIX SHAPE: ", correlation_matrix.shape)

    ########
    # EIGS #
    ########

    print()
    eigs = linalg.eig(correlation_matrix)
    vals = eigs[0]
    vecs = eigs[1]
    print("EIG VALS SHAPE: ", vals.shape)
    print("EIG VECS SHAPE: ", vecs.shape)

    #################
    # PLOT EIG VALS #
    #################

    # plt.plot(vals)
    # plt.show()

    #######
    # PCA #
    #######

    print()
    k = 20;
    images_proj = projection(images_m, vecs, k)
    print("IMAGE PROJ SHAPE: ", images_proj.shape)

    #############
    # SAVE EIGS #
    #############

    print()
    np.save("images-eigs-val.npy", vals)
    np.save("images-eig-vec.npy", vecs)
    print("saved")
