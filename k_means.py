import numpy as np
from tqdm import tqdm

class KMeans:
    """
    Kmeans class.

    Attributes:
        n_cluster: Number of clusters that will be used
        max_iter: Max number of iteration to run while fitting
        random_state: Int to seed numpy randomness with
    """
    def __init__(self, n_cluster, max_iter=100, random_state=123):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.random_state = random_state

    def initialise_centroids(self, X):
        """
        Chooses n_cluster centroids from the data.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)

        Returns:
            Array: Array of length n_cluster, where each element is a randomly
            selected centroid
        """
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_cluster]]
        return centroids

    def compute_centroids(self, X, labels):
        """
        Computes a new set of centroids.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)
            labels: Array of length n_points, where each element is the index of
            the centroid that the point is closest to

        Returns:
            Array: Array of length n_cluster, where each element is a newly
            computed centroid
        """
        centroids = np.zeros((self.n_cluster, X.shape[1]))
        for k in range(self.n_cluster):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X):
        """
        Computes the distance from each point to each centroid.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)

        Returns:
            Array: Array of shape (n_points, n_clusters), where each element is
            the distance between the corresponding point and centroid
        """
        distance = np.zeros((X.shape[0], self.n_cluster))
        for k in range(self.n_cluster):
            row_norm = np.linalg.norm(X - self.centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_clusters(self, X):
        """
        Computes the centroid closest to each point in X.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)

        Returns:
            Array: Array of length n_points, where each element is the index of
            the centroid that the point is closest to
        """
        distance = self.compute_distance(X)
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels=None):
        """
        Computes the sum of squares error for each point with the current
        centroid.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)
        
        Returns:
            Float: Sum of squares error for the passed points and current
            centroids
        """
        if labels is None:
            labels = self.find_closest_clusters(X)
        distance = np.zeros(X.shape[0])
        for k in range(self.n_cluster):
            dist = np.linalg.norm(X[labels == k] - self.centroids[k], axis=1)
            distance[labels == k] = dist
        return np.sum(np.square(distance))
    
    def fit(self, X):
        """
        Calculates centroid for the passed dataset. Runs until either max
        iterations have been run, or the centroids stop changing.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)

        Returns:
            Float: Sum of squares error for the passed points and final
            centroids
        """
        self.centroids = self.initialise_centroids(X)
        for i in tqdm(range(self.max_iter)):
            labels = self.find_closest_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.compute_centroids(X, labels)
            if np.all(old_centroids == self.centroids):
                break
        return self.compute_sse(X)

    def classify_centroids(self, X, class_labels):
        centroid_labels = np.zeros(self.n_cluster)
        closest_clusters = self.find_closest_clusters(X)
        for k in range(self.n_cluster):
            print(class_labels)
            print(class_labels.shape)
            print(type(class_labels[0]))
            input()
            counts = np.bincount(class_labels[closest_clusters == k])
            print(f"{max(counts)} out of {sum(counts)} assigned points.")
            print(f"{max(counts) / sum(counts)}")
            centroid_labels[k] = np.argmax(counts)
        return centroid_labels
    
    def __call__(self, X):
        """
        Call the KMeans object on a set of points.

        Parameters:
            X: Array of points to fit to of shape (n_points, n_dimensions)

        Returns:
            Array: Array of length n_points, where each element is the index of
            the centroid that the point is closest to
        """
        return self.find_closest_cluster(X)
