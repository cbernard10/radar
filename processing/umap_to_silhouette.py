from umap_to_hdbscan import umap_to_hdbscan
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from umap_to_hdbscan import umap_to_hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score


def umap_to_silhouette(cluster_labels, X, min_cluster_size, ignore_label=None, show=False):
    # cluster_labels, silhouette_avg, sample_silhouette_values = umap_to_hdbscan(umap_data,
    #                                                                            min_cluster_size=min_cluster_size,
    #                                                                            save=save)

    silhouette_avg, sample_silhouette_values = silhouette_score(X, cluster_labels), silhouette_samples(X,
                                                                                                       cluster_labels)

    n_clusters = len(list(set(cluster_labels)))

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])

        y_lower = 10
        for i in range(n_clusters):
            if i == ignore_label: continue
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], c=colors, s=1)

        plt.suptitle(
            "Silhouette analysis for HDBSCAN clustering on sample data with min_cluster_size = %d"
            % min_cluster_size,
            fontsize=14,
            fontweight="bold",
        )

        plt.show()

    return cluster_labels, sample_silhouette_values
