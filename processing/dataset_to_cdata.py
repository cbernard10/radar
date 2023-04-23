from glob import glob

import numpy as np
import scipy.io as sio
import os

from .dataset_to_summary import load_summary as ls

ROOT = 'csir/'

def get_datasets():

    """ gets the names of all the datasets """

    import platform
    path = ROOT
    pf = platform.system()

    datasets = glob(path + '*')
    if pf=='Linux':
        datasets = sorted([folder.split('/')[-1] for folder in datasets if
                   folder not in ['..\csir\document', '..\csir\summary.json', '..\csir\make-summary.py',
                                  '..\csir\summary.json.old']])
    elif pf=='Windows':
        datasets = sorted([folder.split('\\')[-1] for folder in datasets if
                   folder not in ['..\csir\document', '..\csir\summary.json', '..\csir\make-summary.py',
                                  '..\csir\summary.json.old']])

    datasets = [ds for ds in datasets if len(ds.split('.')) == 1 and ds[-2] == 'F']

    return datasets

def normalize_cdata(cdata):

    """ normalizes cdata """

    return (cdata - np.median(cdata)) / (
            np.percentile(cdata, 75) - np.percentile(cdata, 25))

def trim_cdata(cdata):

    """ deletes rows containing invalid values """

    new_cdata = []

    if len(cdata.shape) == 3:
        cdata_t = np.transpose(cdata, (1, 0, 2))
    else:
        cdata_t = cdata.T

    for i, line in enumerate(cdata_t):

        all_non_zero = 1
        for cell in line:
            if np.linalg.norm(cell) == 0:
                all_non_zero = 0
                break
        if all_non_zero:
            new_cdata.append(line)

    if len(cdata.shape) == 3:
        return np.transpose(np.array(new_cdata), (1, 0, 2))
    else:
        return new_cdata.T

def dataset_to_cdata(dataset, subsampling=1, normalized=True):

    """ returns cdata for the dataset """

    dataset = dataset.strip()
    paths = sorted(glob(ROOT + dataset + '/' + dataset + '.*.mat'))
    paths = [file for file in paths if file.split('.')[-2] != 'summary']
    dicts = [sio.loadmat(mat_path) for mat_path in paths]
    keys = dicts[0].keys()
    key = 'CData'
    if 'CData' not in keys:
        if 'SingleFrame' not in keys:
            print('fail')
            return -1
        else:
            key = 'SingleFrame'

    matrices = [np.array(dic[key]) for dic in dicts]
    cdata = np.vstack(matrices)

    if dataset[8:10] == 'Sc':
        summ = ls(dataset)
        summ = np.squeeze(summ["PCI"]["ScanIdx"])
        indices1 = np.argwhere(summ == 1)
        indices2 = np.argwhere(summ == 2)
        indices = np.concatenate((indices1, indices2))
        cdata = np.squeeze(cdata[indices])

    cdata = trim_cdata(cdata[::subsampling])

    return normalize_cdata(cdata) if normalized else cdata

def cdata_to_intensity(cdata):

    """ converts cdata to range intensity map """

    I = np.amax(np.real(cdata), -1)
    Q = np.amax(np.imag(cdata), -1)
    res = 10*np.log(10* (I**2 + Q**2))

    return res

def cdata_to_amplitude(cdata):

    """ converts cdata to amplitude map """

    I = np.real(cdata)
    Q = np.imag(cdata)

    return np.linalg.norm( np.concatenate((I, Q), axis=-1) , axis = -1)