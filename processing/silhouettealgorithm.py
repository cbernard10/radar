from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.optimize import minimize, brute
from .dataset_to_cdata import dataset_to_cdata
from .cdata_to_burg import cdata_to_burg
from .burg_to_umap import burg_to_umap
from .umap_to_hdbscan import umap_to_hdbscan
from ..visualisation.visualisation import blue, green
import numpy as np

def silhouette_optimizer(dataset, subsampling=1):
    """ optimiser la silhouette avec scipy.optimize """

    burg = cdata_to_burg(
        dataset_to_cdata(dataset, subsampling = subsampling), dataset, order=6, gamma=0.001
    )

    print(burg.shape)

    def silhouette_loss(x):

        """ x: [min_neighbors, min_dist, min_cluster_size, min_samples] """

        if (x[0] < 0 or x[1] < 0 or x[2] < 0 or x[3] < 0):
            return -1

        X = burg_to_umap(
                burg, int(x[0]), x[1]
            )[0]

        print(X.shape)

        labels = umap_to_hdbscan(
            X, int(x[2]), int(x[3])
        )[1]

        unique = list(set(labels))
        if len(unique) == 1: return -1

        print(labels.shape)

        score = np.abs(1 - silhouette_score(X, labels))

        print(blue(x), green(score))

        return score

    # return minimize(silhouette_loss, [30, 0, 30, 5])

    return brute(silhouette_loss,
                 (slice(10, 50, 10),
                 slice(0, 1, 0.1),
                 slice(10, 50, 10),
                 slice(3, 10, 1)), disp=True, finish=None)