import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from tqdm import tqdm

from .dataset_to_cdata import dataset_to_cdata
from .dataset_to_summary import load_summary
from ..visualisation.visualisation import map_and_scatter, plot_burg
from ..visualisation.visualisation import red

ROOT = 'csir/'

@jit
def burg_reg_jit(signal, N, gamma):

    N = N - 1
    f, b, f_, b_ = signal.copy(), signal.copy(), signal.copy(), signal.copy()
    mus = np.zeros((N,), dtype=np.complex128)
    a = np.zeros(N + 1, dtype=np.complex128)
    a[0] = 1
    P = np.mean(np.abs(signal) ** 2) + 0j

    for n in range(N):

        beta = gamma * (2 * np.pi) ** 2 * (np.arange(0, n) - n) ** 2

        curr_a = a[:n + 1]
        D = np.conj(b[n:-1]) @ f[n + 1:] + 0j
        D = 2 * D / (N - n)
        S = 0j
        if n > 1:
            for k in range(1, n):
                S += beta[k] * curr_a[k] * curr_a[n - k]
            D += 2 * S

        G = np.linalg.norm(f[n + 1:]) ** 2 + np.linalg.norm(b[n:-1]) ** 2 + 0j
        G = G / (N - n)
        S = 0 + 0j
        if n > 0:
            for k in range(0, n):
                S += beta[k] * (np.absolute(curr_a[k])) ** 2
            G += 2 * S

        mu = -D / G

        # A = np.concatenate((a, [0]))
        A = a[:n + 2]
        V = A[::-1]
        a[:n + 2] = A + mu * np.conj(V)

        mus[n] = mu

        f_[1:] = f[1:] + mu * b[:-1]
        b_[1:] = b[:-1] + np.conj(mu) * f[1:]

        f = f_.copy()
        b = b_.copy()

    # print(P, mus)
    return P, mus

def burg_reg_3d_jit(CData, order, gamma, progress=False):
    X, Y, N = CData.shape
    coeffs = np.zeros((X, Y, order), dtype=np.complex128)
    # A_matrix = np.zeros((*coeffs.shape[:2], order), dtype=np.complex128)

    f = lambda x: x
    if progress:
        f = lambda x: tqdm(x)

    for i in f(range(X)):
        for j in range(Y):
            try:
                coeffs[i, j, 0], coeffs[i, j, 1:] = burg_reg_jit(CData[i, j], order, gamma)
            except ZeroDivisionError as _:
                # print('error: division by zero')
                coeffs[i, j, 0], coeffs[i, j, 1:] = 0, 0
                continue

    return coeffs


def trim_cdata(cdata):
    # deletes jth row if any cell of that row is nan
    new = cdata.copy()
    for j in range(cdata.shape[1]):
        for i in range(cdata.shape[0]):
            if np.isnan(cdata[i, j, 0]) or cdata[i, j, 0] == 0j:
                new = np.delete(cdata, j, 1)
                j = j - 1

    print(cdata.shape, new.shape)
    return new


def dataset_to_burg_jit(dataset, order, gamma, subsampling=1, progress=False, save=False, suffix=''):
    try:
        cdata = dataset_to_cdata(dataset, subsampling)

    except IndexError:
        print(red('Could not find mats for this dataset'))
        return None
    else:
        return cdata_to_burg_jit(cdata, dataset, order, gamma, progress, save, suffix)


def cdata_to_burg_jit(cdata, dataset, order, gamma, progress=False, save=False, suffix='', verbose=0):
    currSummary = load_summary(dataset)
    rangeStart = currSummary["PCI"]["Range"]
    ranges = np.array(currSummary["PCI"]["RangeOffset"]) + rangeStart
    times = np.array(currSummary["PCI"]["Time"])

    coeffs = burg_reg_3d_jit(cdata, order, gamma, progress)

    f = lambda x:x
    if verbose:
        f = lambda x:tqdm(x)

    if save:
        for i in f(range(1, order)):
            fig, ax = map_and_scatter(coeffs[:, ::-1, i], ranges, times, 1)
            if not (os.path.isdir(ROOT + dataset + '/burg')):
                os.mkdir(ROOT + dataset + '/burg/')
            im_path = ROOT + dataset + '/burg/' + 'c' + str(i) + suffix + '.png'
            if progress:
                print(im_path)
            plt.savefig(im_path, bbox_inches='tight', dpi=200)
            plt.close(fig)

        fig, ax = show_burg_map(coeffs[:, ::-1, 0], ranges, times, 1)
        im_path = ROOT + dataset + '/burg/' + 'c0' + suffix + '.png'
        if progress:
            print(im_path)
        plt.savefig(im_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

    return coeffs



@njit
def barb_distance(t1, t2):
    n = t1.shape[0]
    A = (n - np.arange(1, n)) / 4
    B = (1 + (np.abs((t1[1:] - t2[1:]) / (1 - t1[1:] * np.conj(t2[1:]))))) / (
            1 - (np.abs((t1[1:] - t2[1:]) / (1 - t1[1:] * np.conj(t2[1:])))))
    C = np.sum(A * np.log(B) ** 2)
    return np.abs(n * np.log(t2[0] / t1[0]) ** 2 + C)

# if __name__ == '__main__':
#    apply_burg_get_datasets(int(sys.argv[1]), sys.argv[2] if len(sys.argv) == 3 else '00_005_TTrFA')

#  robocopy ../csir/ ../csirexplorer-next/public/assets/images *.png /E
