import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..visualisation.visualisation import map_and_scatter, plot_burg, yellow
from .signal_to_burg import burg_reg, burg_spectrum

ROOT = 'csir/'

def burg_reg_3d(CData, order, gamma, progress=False):

    """ performs the burg algorithm on 3D radar data """

    if len(CData.shape) < 3:
        print(yellow('cannot compute reflection coefficients for iq data'))
        return np.zeros((CData.shape[0], CData.shape[1], order))
    X, Y, N = CData.shape
    coeffs = np.zeros((X, Y, order), dtype=np.complex128)

    f = lambda x: x
    if progress:
        f = lambda x: tqdm(x)

    for i in f(range(X)):
        for j in range(Y):
            try:
                coeffs[i, j, 0], coeffs[i, j, 1:], _, _ = burg_reg(CData[i, j], order-1, gamma)
            except ZeroDivisionError as _:

                coeffs[i, j, 0], coeffs[i, j, 1:] = 0, 0
                continue

    return coeffs


def trim_cdata(cdata):

    new = cdata.copy()
    for j in range(cdata.shape[1]):
        for i in range(cdata.shape[0]):
            if np.isnan(cdata[i, j, 0]) or cdata[i, j, 0] == 0j:
                new = np.delete(cdata, j, 1)
                j = j - 1

    print(cdata.shape, new.shape)
    return new

def cdata_to_burg(cdata, dataset, order, gamma, progress=False, save=False, suffix='', verbose=0):

    """ erforms the burg algorithm on 3D radar data and saves the result as png images """

    coeffs = burg_reg_3d(cdata, order, gamma, progress)

    f = lambda x:x
    if verbose:
        f = lambda x:tqdm(x)

    if save:
        for i in f(range(1, order)):
            fig, ax = map_and_scatter(coeffs[:, ::-1, i], ranges, times, 1)
            if not (os.path.isdir('csir/' + dataset + '/burg')):
                os.mkdir(ROOT + dataset + '/burg/')
            im_path = ROOT + dataset + '/burg/' + 'c' + str(i) + suffix + '.png'
            if progress:
                print(im_path)
            plt.savefig(im_path, bbox_inches='tight', dpi=200)
            plt.close(fig)

        fig, ax = plot_burg(coeffs[:, ::-1, 0], ranges, times, 1)
        im_path = ROOT + dataset + '/burg/' + 'c0' + suffix + '.png'
        if progress:
            print(im_path)
        plt.savefig(im_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

    return coeffs

def cdata_to_burg_spectrum(cdata, azimuth, order, nfft=64, gamma=0, normalize=True):

    """ computes the PSD from the reflection coefficients """

    cdata_az = cdata[azimuth]
    spectrum = np.zeros((cdata_az.shape[0], nfft))

    for r in range(cdata_az.shape[0]):

        res = 10*np.log10(np.array(np.fft.fftshift(burg_spectrum(cdata_az[r], order=order, NFFT=nfft, gamma=gamma))))
        spectrum[r] = res

    if normalize:
        spectrum = spectrum - np.min(spectrum)
        spectrum = spectrum / np.max(spectrum)

    return spectrum