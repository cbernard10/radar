import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
from scipy.special import gamma, kv
from scipy.optimize import minimize_scalar
# from dataset_to_cdata import cdata_to_amplitude

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from tqdm import tqdm
from ..visualisation.visualisation import green, red, yellow

from multiprocessing import Pool
import os
import time

def DBSCANCDATA(cdata, eps, min_samples):

    c=cdata
    iq = np.concatenate(
        (
            np.real(c), np.imag(c)
        ), axis=-1
    )
    samples = np.reshape(iq, (iq.shape[0] * iq.shape[1], iq.shape[2]))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(samples)
    iqClustered = np.reshape(clustering.labels_, (iq.shape[0], iq.shape[1]))
    return iqClustered.T

def DBSCAN_WINDOW(signal, idx, guard_cells, ref_cells, eps, min_samples):

    lagging_window = signal[idx - (guard_cells + ref_cells +1): idx - guard_cells - 1]
    leading_window = signal[idx + guard_cells : idx + guard_cells + ref_cells]

    reference_cells = np.concatenate((lagging_window, leading_window), axis=0)
    # print(f"ref cells shape", reference_cells.shape)

    data_points = np.concatenate((np.real(reference_cells), np.imag(reference_cells)), axis=-1)
    # print('dp shape', data_points.shape, data_points[0])

    # print(data_points)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_points)

    # print('c done')
    outliers = data_points[clustering.labels_ == -1]
    normal = data_points[clustering.labels_ != -1]

    print('normal', clustering.labels_)

    S = outliers.shape[0]
    N = signal.shape[0]
    sq_normal = [ z[0]**2 + z[1]**2 for z in normal ]

    Z_DBSCAN = np.mean(sq_normal)

    return Z_DBSCAN

def GET_ZDBSCAN(matrix, n_control, n_guard, eps, min_samples):

    matrix = np.transpose(matrix, (1, 0, 2))
    zdbscan = np.zeros((matrix.shape[0], matrix.shape[1]))

    for i in range(matrix.shape[0]):
        for j in range(n_control + n_guard + 1, matrix.shape[1] - n_guard - n_control):

            # print(i,j)
            signal = matrix[i]
            # print(signal.shape)
            zdbscan[i][j] = DBSCAN_WINDOW(matrix[i], j, n_guard, n_control, 3, 3)

    return np.transpose(zdbscan, (1, 0, 2))

def GEN_SIGNAL(c, v, size):
    # -*- coding: utf-8 -*-
    # np.random.seed((os.getpid() * int(time.time())) % 123456789)
    np.random.seed()
    class k_gen2(rv_continuous):
        def _pdf(self, x, c, v):
            return 2*c / gamma(v) * np.sqrt(c*x)**(v-1) * kv(v-1, 2*np.sqrt(c*x))

    k = k_gen2(name='k', a=0, b=np.inf)

    return k.rvs(c, v, size=size)

def GEN_CLUTTER(c, v, size):

    length = np.prod(size)
    print(green(f"computing {length} coefficients"))

    with Pool(16) as p:
        res = p.starmap(GEN_SIGNAL, [(c, v, (1,))]*length)

    return np.reshape(res, size)


def GEN_SIGNAL_IMAP(args):

    c, v, size = args
    return GEN_SIGNAL(c, v, size)


def COMPUTE_HISTOGRAMS_FOR_SCALE_RANGE(batch_size=5000, min_scale=0.1, max_scale=30, step=0.01):

    from multiprocessing import Pool

    signals = []

    print(green('computing signals...'))
    scales = np.arange(min_scale, max_scale, step)
    print(list(zip([batch_size]*scales.shape[0], scales, scales)))

    with Pool(16) as p:
        res = p.map(GEN_SIGNAL_IMAP, list(zip(scales, scales, [batch_size]*scales.shape[0])))

    print(res)

    print(green('computing histograms...'))
    hists = [np.histogram(r, bins=50, range=(0.01, 30))[0] for r in tqdm(res)]

    return hists

def MOM12KDIST(avg, var):

    def loss_shape(v):
        return (
                (gamma(1+0.5) ** 2 * gamma(v + 0.5)**2)/(v * gamma(v)**2) - avg**2 / var
        ) ** 2

    v = minimize_scalar(loss_shape).x
    c = avg * gamma(v) / (np.sqrt(np.pi) * gamma(v + 0.5))

    return v, c



""" minimiser loss f de alpha tq pour pfa donné,  """

""" pour une image radar simulée : 
 - calculer Z
 - estimer v avec méthode des moments
 - pour chaque v :
        trouver le vrai pfa en fonction des pfa = [1e-3, 1e-4, 1e-5], moyenne des """