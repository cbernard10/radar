# -*- coding: utf-8 -*-

""" reference : https://dspace.mit.edu/bitstream/handle/1721.1/123125/1128277010-MIT.pdf?sequence=1&isAllowed=y """
"""
Utilizing I/Q Data to Enhance Radar Detection and Accuracy Metrics, Alexandria Velez
"""

# from __future__ import division
import numpy as np
import scipy as sp
from scipy.special import binom
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from math import factorial
from scipy.special import gamma

#import pynverse

def CFAR2D(data):
    """ 2.1 """
    kernel = np.ones((9, 9))
    kernel[2:7, 2:7] = 0
    noise_matrix = sp.signal.convolve2d(data, kernel, mode="same")

    return (10*np.log10( data / noise_matrix) > 13)

def CACFAR2D(matrix,neigh_num,guard_num,PFA):

    matrix = np.absolute(matrix)
    detected = np.zeros_like(matrix,dtype=bool)

    pad_matrix = np.pad(matrix,neigh_num,'symmetric')

    start = (neigh_num,neigh_num)
    end = (matrix.shape[0]+neigh_num,matrix.shape[1]+neigh_num)

    for i in tqdm(range(start[0],end[0])):
        for j in range(start[1],end[1]):

            A = pad_matrix[i-neigh_num:1+i+neigh_num,j-neigh_num:j+1+neigh_num]
            B = pad_matrix[i-guard_num:1+i+guard_num,j-guard_num:j+1+guard_num]
            B_ = np.zeros_like(A)
            B_[neigh_num-guard_num:-(neigh_num-guard_num),neigh_num-guard_num:-(neigh_num-guard_num)] = B

            neighs = A-B_
            neighs = np.trim_zeros(neighs.ravel())

            CUT = pad_matrix[i,j]

            detected[i-neigh_num,j-neigh_num] = (CUT > np.mean(neighs)*(PFA**(-1/(2* ( (neigh_num + 1)**2 - (guard_num + 1)**2) )) - 1))
#
    return detected

def _get_control_cells(array, i, n_control, n_guard):
    return np.concatenate( (
        array[i - n_control - n_guard +1: i - n_guard+1],
        array[i + n_guard: i + n_control + n_guard]
    ) )

def _CACFAR(doppler_data, n_control, n_guard, PFA):

    n_ranges = doppler_data.shape[0]
    cacfar_matrix = np.zeros((n_ranges, doppler_data.shape[1] - 2*(n_control + n_guard)))

    for i in tqdm(range(n_ranges)):
        for j in range(cacfar_matrix.shape[1]):
            c_cells = _get_control_cells(doppler_data[i], j+n_control+n_guard, n_control, n_guard)
            mean = np.sum(c_cells)
            control = mean * (PFA**(-1/(2*n_control)) - 1)
            cacfar_matrix[i, j] = control

    return doppler_data[:, n_control+n_guard: - n_control - n_guard] > cacfar_matrix

# def CACFAR(doppler_data, n_control, n_guard, PFA):
#
#     n_ranges = doppler_data.shape[0]
#     cacfar_matrix = np.zeros((n_ranges, doppler_data.shape[1] - 2*(n_control + n_guard)))
#
#     for i in tqdm(range(n_ranges)):
#         for j in range(cacfar_matrix.shape[1]):
#             c_cells = _get_control_cells(doppler_data[i], j+n_control+n_guard, n_control, n_guard)
#             mean = np.sum(c_cells)
#             control = mean * (PFA**(-1/(2*n_control)) - 1)
#             cacfar_matrix[i, j] = control
#
#     return doppler_data[:, n_control+n_guard: - n_control - n_guard] > cacfar_matrix

def CACFAR(rd_map, n_control, n_guard, PFA):
    """
    Detect peaks with CFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    mask = np.zeros_like(rd_map)

    for k in tqdm(range(rd_map.shape[0])):

        x = rd_map[k]
        num_cells = x.size
        num_train_half = round(n_control / 2)
        num_guard_half = round(n_guard / 2)
        num_side = num_train_half + num_guard_half

        alpha = n_control * (PFA ** (-1 / n_control) - 1)  # threshold factor

        for i in range(num_side, num_cells - num_side):

            if i != i - num_side + np.argmax(x[i - num_side:i + num_side + 1]):
                continue

            sum1 = np.sum(x[i - num_side:i + num_side + 1])
            sum2 = np.sum(x[i - num_guard_half:i + num_guard_half + 1])
            p_noise = (sum1 - sum2) / n_control
            threshold = alpha * p_noise

            mask[k, i] = x[i] > threshold

    return mask

def GOCFAR_PFA_calc(T, M):
    """ source: Detectability Loss Due to "Greatest Of" Selection in a Cell-Averaging CFAR, eq (7) """
    """ Computes the PFA from a threshold T in dB and the window size M """

    return 2*( (1+T/M)**(-M) - (2 + T/M)**(-M) * sum( [binom(M-1-k, k) * (2+T/M) ** (-M) for k in range(M)] ) )

def get_GOCFAR_threshold(PFA, M):

    """ computes the required threshold for a set PFA and window size M with gradient descent """

    def loss(_T):
        _PFA = GOCFAR_PFA_calc(_T, M)
        return np.abs(_PFA - PFA)

    res = minimize_scalar(loss)
    return res['x']

def GOCFAR(rd_map, n_control, n_guard, PFA):
    """
    Detect peaks with GOCFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    mask = np.zeros_like(rd_map)

    for k in tqdm(range(rd_map.shape[0])):

        x = rd_map[k]
        num_cells = x.size
        n_side = n_control + n_guard

        alpha = n_control * get_GOCFAR_threshold(PFA, n_control)  # threshold factor

        for i in range(n_side, num_cells - n_side):

            if i != i - n_side + np.argmax(x[i - n_side:i + n_side + 1]):
                continue

            ref_cells_left = x[i - n_side:i - n_guard + 1]
            ref_cells_right = x[i + n_guard:i + n_guard + n_control + 1]
            ref_cells = np.concatenate((ref_cells_left, ref_cells_right))
            p_noise = max(ref_cells) / n_control
            threshold = alpha * p_noise

            mask[k, i] = x[i] > threshold

    return mask


def SOCFAR_PFA_calc(T, M):
    """ source: Detectability Loss Due to "Greatest Of" Selection in a Cell-Averaging CFAR, eq (7) """
    """ Computes the PFA from a threshold T in dB and the window size M """

    return 2*( (2 + T/M)**(-M) * sum( [binom(M-1+k, k) * (2+T/M) ** (-k) for k in range(M)] ) )

def get_SOCFAR_threshold(PFA, M):

    """ computes the required threshold for a set PFA and window size M with gradient descent """

    def loss(_T):
        _PFA = SOCFAR_PFA_calc(_T, M)
        return np.abs(_PFA - PFA)

    res = minimize_scalar(loss)
    return res['x']

def SOCFAR(rd_map, n_control, n_guard, PFA):
    """
    Detect peaks with SOCFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    mask = np.zeros_like(rd_map)

    for k in tqdm(range(rd_map.shape[0])):

        x = rd_map[k]
        num_cells = x.size
        n_side = n_control + n_guard

        alpha = n_control * get_SOCFAR_threshold(PFA, n_control)  # threshold factor

        for i in range(n_side, num_cells - n_side):

            if i != i - n_side + np.argmax(x[i - n_side:i + n_side + 1]):
                continue

            ref_cells_left = x[i - n_side:i - n_guard + 1]
            ref_cells_right = x[i + n_guard:i + n_guard + n_control + 1]
            ref_cells = np.concatenate((ref_cells_left, ref_cells_right))
            p_noise = min(ref_cells) / n_control
            threshold = alpha * p_noise

            mask[k, i] = x[i] > threshold

    return mask


def OSCFAR_PFA_calc(T, M, K):
    """ source: Detectability Loss Due to "Greatest Of" Selection in a Cell-Averaging CFAR, eq (7) """
    """ Computes the PFA from a threshold T in dB and the window size M """

    A = factorial(M) / factorial(M-K)
    B = gamma(T + M - K + 1)/gamma(T + M + 1)

    return A*B

def get_OSCFAR_threshold(PFA, M, K):

    """ computes the required threshold for a set PFA and window size M with gradient descent """

    def loss(_T):
        _PFA = OSCFAR_PFA_calc(_T, M, K)
        return np.abs(_PFA - PFA)

    res = minimize_scalar(loss)
    return res['x']

def OSCFAR(rd_map, n_control, n_guard, K, PFA):
    """ http://www.eng.tau.ac.il/~nadav/pdf-files/perforos.pdf """

    mask = np.zeros_like(rd_map)

    for k in tqdm(range(rd_map.shape[0])):

        x = rd_map[k]
        num_cells = x.size
        n_side = n_control + n_guard

        alpha = n_control * get_OSCFAR_threshold(PFA, 2*n_control - 2*n_guard, K)  # threshold factor

        for i in range(n_side, num_cells - n_side):

            if i != i - n_side + np.argmax(x[i - n_side:i + n_side + 1]):
                continue

            ref_cells_left = x[i - n_side:i - n_guard + 1]
            ref_cells_right = x[i + n_guard:i + n_guard + n_control + 1]

            ref_cells = np.concatenate((ref_cells_left, ref_cells_right))
            ref_cells = np.sort(ref_cells)
            noise_and_clutter_level = ref_cells[K]

            threshold = alpha * noise_and_clutter_level

            mask[k, i] = x[i] > threshold

    return mask

# def GCMLDCFAR(rd_map, n_control, n_guard, PFA):


""" TPCFAR,  LNCFAR, KCFAR, Bilateral CFAR, ISCFAR, IBCFAR, ANDCFAR, ORCFAR """