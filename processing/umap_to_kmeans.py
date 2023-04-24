from os import unlink

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

from ..visualisation.visualisation import convert_to_jpeg

ROOT = 'csir/'

def umap_to_kmeans(umap_data, dataset, filename, s=1, subFolder=''):

    try:
        X = umap_data[0].embedding_

        kmeans = KMeans(n_clusters=2).fit(X)
        cluster_labels = kmeans.labels_

        f = plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=s)
        im_path = ROOT + dataset + '/burg/umap/' + subFolder + '/' + filename
        plt.savefig(im_path + '.png', bbox_inches='tight', dpi=200)
        plt.close(f)
        convert_to_jpeg(im_path)
        return cluster_labels
    except TypeError as e:
        print(repr(e))
        return None
