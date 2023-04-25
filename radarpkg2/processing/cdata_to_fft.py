import numpy as np
from .dataset_to_cdata import dataset_to_cdata
from .dataset_to_summary import load_summary
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def red(text):
    return "\033[1;91m " + text + "\033[0m"

ROOT = 'csir/'

def normalize_data(data):

    """ interquartile range normalization """

    return (data - np.median(data)) / (
            np.percentile(data, 75) - np.percentile(data, 25))

def cdata_to_fft(cdata, azimuth, NFFT = 64, normalize=True):

    """ performs the FFT on radar data """

    iq_data = cdata[azimuth]
    ft = np.abs(np.fft.fftshift(np.fft.fft(iq_data, axis = -1, n=NFFT)))
    if normalize:
        ft = ft-np.min(ft)
        ft = ft/np.max(ft)
    return np.concatenate((ft[round(ft.shape[0]/2): , :], ft[:round(ft.shape[0]/2), :]))

    # iq_data = np.reshape(cdata, (cdata.shape[0] * cdata.shape[2], cdata.shape[1]))
    # return np.fft.fftshift(np.fft.fft(iq_data, axis = -1)).T

def dataset_to_fft(dataset):
    rd = cdata_to_fft(dataset)
    f_rd = np.sqrt(np.absolute(rd))
    fig = plt.figure()
    plt.imshow(f_rd[:, ::-1], cmap='coolwarm', aspect='auto', interpolation='nearest')
    plt.show(block=False)
    plt.savefig(ROOT + dataset + '/' + dataset + '_FFT.png', dpi=300)
    plt.close(fig)


def datasets_to_fft(start=None):
    summary = load_summary()
    root = ROOT
    paths = glob(root + '*')
    paths = [path for path in paths if len(path.split('.')) == 3]
    start_from = paths.index(root + start)
    paths = paths[start_from:]
    for path in tqdm(paths):
        dataset = path.split('\\')[-1]

        if not summary[dataset]['ignore']:

            dataset_to_fft(dataset)

        else:
            print(red('dataset ignored'))