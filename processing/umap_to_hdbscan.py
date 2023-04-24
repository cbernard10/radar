import hdbscan
import matplotlib.pyplot as plt
from PIL import Image
from os import unlink

from ..visualisation.visualisation import convert_to_jpeg

ROOT = 'csir/'

def umap_to_hdbscan(umap_data, min_cluster_size=100, min_samples=None, s=1, subFolder='', save=0, dataset=None,
                    filename=None, verbose=0):
    # try:
        X = umap_data
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), allow_single_cluster=True,
                                    min_samples=min_samples)
        if verbose: print('clustering...')
        cluster_labels = clusterer.fit_predict(X)
        if save:
            f = plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=s)
            print(ROOT, dataset, subFolder, filename)
            im_path = ROOT + dataset + '/burg/umap/' + subFolder + '/' + filename
            # plt.show()

            plt.savefig(im_path + '.png', bbox_inches='tight', dpi=200)
            plt.close(f)
            convert_to_jpeg(im_path)

        return X, cluster_labels
    # except TypeError as e:
    #     print(repr(e))
    #     return None
    # except ValueError as e:
    #     print(repr(e))
    #     return None


def plot_hdbscan(embedding, labels, s=1):

    X = embedding
    f = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=s)
    plt.show()
