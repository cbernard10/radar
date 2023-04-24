import numpy as np
from numba import jit, njit
from spectrum import arma2psd

@jit
def burg_reg(X, order, gamma=0):

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """ regularized burg algorithm of order <order> for the signal X with coefficient <gamma>                                   """
    """ returns average power P, reflection coefficients mus, autocorrelation coefficents matrix a, white noise variance rho    """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    N = X.shape[0]
    f, b, f_, b_ = X.copy(), X.copy(), X.copy(), X.copy()
    mus = np.zeros((order,), dtype=np.complex128)
    a = np.zeros(order + 1, dtype=np.complex128)
    a[0] = 1
    P = np.mean(np.abs(X) ** 2) + 0j
    rho = sum(np.abs(X) ** 2) / X.shape[0]
    den = rho * 2. * N

    for n in range(order):

        beta = gamma * (2 * np.pi) ** 2 * (np.arange(0, n) - n) ** 2

        curr_a = a[:n + 1]
        D = np.conj(b[n:-1]) @ f[n + 1:] + 0j
        D = 2 * D / (order - n)
        S = 0j
        if n > 1:
            for k in range(1, n):
                S += beta[k] * curr_a[k] * curr_a[n - k]
            D += 2 * S

        G = np.linalg.norm(f[n + 1:]) ** 2 + np.linalg.norm(b[n:-1]) ** 2 + 0j
        G = G / (order - n)
        S = 0 + 0j
        if n > 0:
            for k in range(0, n):
                S += beta[k] * (np.absolute(curr_a[k])) ** 2
            G += 2 * S

        mu = -D / G
        kp = mu
        temp = 1. - np.abs(kp)**2
        rho = temp * rho

        # A = np.concatenate((a, [0]))
        A = a[:n + 2]
        V = A[::-1]
        a[:n + 2] = A + mu * np.conj(V)

        mus[n] = mu

        f_[1:] = f[1:] + mu * b[:-1]
        b_[1:] = b[:-1] + np.conj(mu) * f[1:]

        f = f_.copy()
        b = b_.copy()

    return P, mus, a[1:], rho

def burg_reg_nojit(X, order, gamma=0):

    N = X.shape[0]
    f, b, f_, b_ = X.copy(), X.copy(), X.copy(), X.copy()
    mus = np.zeros((order,), dtype=np.complex128)
    a = np.zeros(order + 1, dtype=np.complex128)
    a[0] = 1
    P = np.mean(np.abs(X) ** 2) + 0j
    rho = sum(np.abs(X) ** 2) / X.shape[0]
    den = rho * 2. * N

    for n in range(order):

        beta = gamma * (2 * np.pi) ** 2 * (np.arange(0, n) - n) ** 2

        curr_a = a[:n + 1]
        D = np.conj(b[n:-1]) @ f[n + 1:] + 0j
        D = 2 * D / (order - n)
        S = 0j
        if n > 1:
            for k in range(1, n):
                S += beta[k] * curr_a[k] * curr_a[n - k]
            D += 2 * S

        G = np.linalg.norm(f[n + 1:]) ** 2 + np.linalg.norm(b[n:-1]) ** 2 + 0j
        G = G / (order - n)
        S = 0 + 0j
        if n > 0:
            for k in range(0, n):
                S += beta[k] * (np.absolute(curr_a[k])) ** 2
            G += 2 * S

        mu = -D / G
        kp = mu
        temp = 1. - np.abs(kp)**2
        rho = temp * rho

        # A = np.concatenate((a, [0]))
        A = a[:n + 2]
        V = A[::-1]
        a[:n + 2] = A + mu * np.conj(V)

        mus[n] = mu

        f_[1:] = f[1:] + mu * b[:-1]
        b_[1:] = b[:-1] + np.conj(mu) * f[1:]

        f = f_.copy()
        b = b_.copy()

    return P, mus, a[1:], rho

def burg_spectrum(X, order, NFFT, gamma=0):

    """ power spectral density from the prediction coefficients computed by the burg algorithm """

    P, ref, A, rho = burg_reg(X, order, gamma)
    psd = arma2psd(A, None, rho, 1, NFFT=NFFT)
    return psd