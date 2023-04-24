import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

from dataset_to_cdata import dataset_to_cdata
from dataset_to_summary import load_summary
from visualisation import map_and_scatter
from visualisation import red

from dataset_to_burg import trim_cdata

ROOT = 'csir/'

def burg_reg_multi(signal):
    return burg_reg(signal, 6, 0)[0]


def burg_reg_3d_parallel(CData, order, gamma):
    length = CData.shape[0] * CData.shape[1]
    data = np.reshape(CData, (length, CData.shape[2]))

    with Pool() as p:
        # result = p.map(partial(partial(burg_reg, N=order), gamma=gamma), data)
        result = p.starmap(burg_reg, zip(data, [order] * length, [gamma] * length))

    result = np.array(result)
    print(result.shape)
    return result.reshape((CData.shape[0], CData.shape[1], order))


def burg_reg_3d_parallel_jit(CData, order, gamma):
    length = CData.shape[0] * CData.shape[1]
    data = np.reshape(CData, (length, CData.shape[2]))

    with Pool() as p:
        # result = p.map(partial(partial(burg_reg, N=order), gamma=gamma), data)
        result = p.starmap(burg_reg, zip(data, [order] * length, [gamma] * length))

    result = np.array(result)
    print(result.shape)
    return result.reshape((CData.shape[0], CData.shape[1], order))


def burg_reg_parallel(signal, N, gamma, result, index):
    result[index] = burg_reg(signal, N, gamma)


@jit
def burg_reg_jit(signal, order, gamma):
    N = order - 1
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
        S = 0j  # temp
        if n > 1:
            for k in range(1, n):
                S += beta[k] * curr_a[k] * curr_a[n - k]
            D += 2 * S

        G = np.linalg.norm(f[n + 1:]) ** 2 + np.linalg.norm(b[n:-1]) ** 2 + 0j
        G = G / (N - n)
        S = 0j
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


# @njit
def burg_reg(signal, N, gamma):
    # burg algorithm on a complex signal

    N = N - 1
    f, b, f_, b_ = signal.copy(), signal.copy(), signal.copy(), signal.copy()
    mus = np.zeros((N,), dtype=np.complex128)
    a = np.array([1])
    P = [np.mean(np.abs(signal) ** 2)]
    buffer = []

    for n in range(N):

        buffer += [signal]
        beta = gamma * (2 * np.pi) ** 2 * (np.arange(0, n) - n) ** 2

        D = np.conj(b[n:-1]) @ f[n + 1:]
        D = 2 * D / (N - n)
        S = 0
        if n > 1:
            for k in range(1, n):
                S += beta[k] * a[k] * a[n - k]
            D += 2 * S

        G = np.linalg.norm(f[n + 1:]) ** 2 + np.linalg.norm(b[n:-1]) ** 2
        G = G / (N - n)
        S = 0
        if n > 0:
            for k in range(0, n):
                S += beta[k] * np.absolute(a[k]) ** 2
            G += 2 * S

        mu = -D / G

        A = np.concatenate((a, [0]))
        V = A[::-1]
        a = A + mu * np.conj(V)

        mus[n] = mu.copy()

        f_[1:] = f[1:] + mu * b[:-1]
        b_[1:] = b[:-1] + np.conj(mu) * f[1:]

        f = f_.copy()
        b = b_.copy()

    # print(P, mus)
    return np.concatenate((P, mus))

def dataset_to_burg_parallel_jit(dataset, order, gamma, subsampling=1, save=False, suffix='', progress=False):
    try:
        cdata = dataset_to_cdata(dataset, subsampling)

    except IndexError:
        print(red('Could not find mats for this dataset'))
        return None
    else:

        currSummary = load_summary(dataset)
        rangeStart = currSummary["PCI"]["Range"]
        ranges = np.array(currSummary["PCI"]["RangeOffset"]) + rangeStart
        times = np.array(currSummary["PCI"]["Time"])

        coeffs = burg_reg_3d_parallel(cdata, order, gamma)

        if save:
            for i in range(1, order):
                print(i)
                fig, ax = map_and_scatter(coeffs[:, ::-1, i], ranges, times, 1)
                if not (os.path.isdir(ROOT + dataset + '/burg')):
                    os.mkdir(ROOT+ dataset + '/burg/')
                im_path = ROOT + dataset + '/burg/' + 'c' + str(i) + suffix + '.png'
                plt.savefig(im_path, bbox_inches='tight', dpi=200)
                plt.close(fig)
                # im = Image.open(im_path)
                # im.convert('RGB').save(im_path.split('.')[0] + 'jpg', 'JPEG')

            f = plt.figure()
            plt.imshow(np.real(coeffs[..., 0].T[::-1]), cmap='gray', aspect='auto', interpolation='none')
            plt.savefig(
                ROOT + dataset + '/burg/' + 'c0' + '.png', bbox_inches='tight', dpi=200)
            plt.close(f)

        return coeffs


def dataset_to_burg_parallel(dataset, order, gamma, subsampling=1, save=False, suffix='', progress=False):
    try:
        cdata = dataset_to_cdata(dataset, subsampling)

    except IndexError:
        print(red('Could not find mats for this dataset'))
        return None
    else:

        currSummary = load_summary(dataset)
        rangeStart = currSummary["PCI"]["Range"]
        ranges = np.array(currSummary["PCI"]["RangeOffset"]) + rangeStart
        times = np.array(currSummary["PCI"]["Time"])

        coeffs = burg_reg_3d_parallel(cdata, order, gamma)

        if save:
            for i in range(1, order):
                print(i)
                fig, ax = map_and_scatter(coeffs[:, ::-1, i], ranges, times, 1)
                if not (os.path.isdir(ROOT + dataset + '/burg')):
                    os.mkdir(ROOT + dataset + '/burg/')
                im_path = ROOT + dataset + '/burg/' + 'c' + str(i) + suffix + '.png'
                plt.savefig(im_path, bbox_inches='tight', dpi=200)
                plt.close(fig)
                # im = Image.open(im_path)
                # im.convert('RGB').save(im_path.split('.')[0] + 'jpg', 'JPEG')

            f = plt.figure()
            plt.imshow(np.real(coeffs[..., 0].T[::-1]), cmap='gray', aspect='auto', interpolation='none')
            plt.savefig(
                ROOT + dataset + '/burg/' + 'c0' + '.png', bbox_inches='tight', dpi=200)
            plt.close(f)

        return coeffs


# from dataset_to_burg import barb_distance
