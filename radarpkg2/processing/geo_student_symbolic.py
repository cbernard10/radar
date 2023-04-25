# -*- coding: utf-8 -*-

import numpy as np
# from scipy.special import polygamma
from sympy import symbols, factorial, polygamma, diff

mu, sigma, v = symbols('mu,sigma,v')
gvv, dgvv = symbols('gvv, dgvv')

def polygamma_stirling(n, x):
    return polygamma(n, x)

def g11(v, sigma):
    return (v + 1) / (v + 3) * sigma ** (-2)


def g22(v, sigma):
    return (2*v) / (sigma**2 * (v+3))

def g33(v, sigma):
    return gvv

def g23(v, sigma):
    return -2/(sigma*(v+1)*(v+3))

def g32(v, sigma):
    return g23(v, sigma)

def metric(v, sigma):
    return np.array([
        [g11(v, sigma), 0, 0],
        [0, g22(v, sigma), g23(v, sigma)],
        [0, g32(v, sigma), g33(v, sigma)]
    ])

def det(v, sigma):
    return np.linalg.det(metric(v, sigma))

def inverse_metric(v, sigma):
    det = g11(v, sigma) * (g33(v, sigma) * g22(v, sigma) - g23(v, sigma) * g32(v, sigma))

    return np.array([
        [((v+3) * sigma**2)/(v+1), 0, 0],
        [0, g11(v, sigma)*g33(v, sigma) / det, -g11(v, sigma)*g23(v, sigma) / det],
        [0, -g11(v, sigma)*g23(v, sigma) / det, g11(v, sigma)*g22(v, sigma) / det]
    ])

#%%

def dg(a, b, c, v, sigma):
    if c == 1:
        if (a == 0 and b == 0):
            return -2 / (sigma ** 3) * (v + 1) / (v + 3)

        elif (a == 1 and b == 1):
            return -4 / (sigma**3) * (1-(3/v+3))

        elif (a == 1 and b == 2 or a == 2 and b == 1):
            return 1/(sigma**2) * (1/(v+1) - 1/(v+3))

        else:
            return 0

    elif c == 2:
        if (a == 0 and b == 0):
            return 2 / (sigma ** 2 * (v + 3) ** 2)

        elif (a == 1 and b == 1):
            return 6 / ((v + 3) ** 2 * sigma ** 2)

        elif (a == 2 and b == 2):
            return dgvv
            return -(polygamma_stirling(2, (v + 1) / 2)/4 - polygamma_stirling(2, v / 2)/4)/2 + 5/(6*v) - 1/(v+1)**2 + 1/(6*(v+3)**2)

        elif (a == 2 and b == 1 or a == 1 and b == 2):
            return 1/sigma * (1/(v+1)**2 - 1/(v+3)**2)

        else:
            return 0

    else:
        return 0

def christ(i, j, k, v, sigma):

    GI = inverse_metric(v, sigma)
    sum = 0
    for l in range(3):
        sum +=  GI[i, l] * (dg(j, l, k, v, sigma) + dg(k, l, j, v, sigma) - dg(j, k, l, v, sigma))

    return sum/2

