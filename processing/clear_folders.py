import shutil
import os

def clear_burg_folder(dataset):
    root = 'csir/'
    if os.path.isdir(root + dataset + '/burg'):
        shutil.rmtree(root + dataset + '/burg')

def clear_umap_folder(dataset):
    root = 'csir/'
    shutil.rmtree(root + dataset + '/burg/umap')
