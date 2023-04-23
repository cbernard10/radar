from radarpkg.processing.dataset_to_cdata import get_datasets
from radarpkg.processing.dataset_to_cdata import dataset_to_cdata
from radarpkg.processing.cdata_to_burg import cdata_to_burg
from radarpkg.processing.burg_to_umap import burg_to_umap, umap_to_images
from radarpkg.processing.umap_to_hdbscan import umap_to_hdbscan, plot_hdbscan
from radarpkg.processing.clear_folders import clear_burg_folder, clear_umap_folder

import os
from radarpkg.visualisation.visualisation import green, yellow, save_burg_maps

def process(subsampling= 1, order=6, gamma= 0.005, n_neighbors=100, min_dist= 0, low_memory= True, min_cluster_size=100, min_samples=10,
            start_from = '00_005_TTrFA'):

    dss = get_datasets()

    skip = 1

    for i, ds in enumerate(dss):

        if ds == start_from:
            skip = 0

        if skip:
            continue

        clear_burg_folder(ds)
        print(yellow(f'computing for {ds}'))

        c = dataset_to_cdata(ds, subsampling=subsampling)
        if len(c.shape) < 3:
            continue
        b = cdata_to_burg(c, ds, order, gamma)
        save_burg_maps(b, ds)
        print(yellow('saved burg'))
        u = burg_to_umap(b, n_neighbors, min_dist, 'euclidean', low_memory=low_memory)
        umap_to_images(u, ds, 'u', 1)
        print(yellow('saved umap'))
        h = umap_to_hdbscan(u, min_cluster_size, min_samples, save=1, filename='hdbscan', dataset=ds)

        if i % 10 == 0:

            os.chdir("csir")
            os.system("node make-summary.js")
            os.chdir("..")
            os.system("rsync -a --include '*/' --include '*.jpg' --exclude '*' csir/ csirexplorer-next/public/assets/images/")
            os.system("rsync -a --include '*/' --include '*.png' --exclude '*' csir/ csirexplorer-next/public/assets/images/")
            print(green(f'iteration {i}, autosaved'))

