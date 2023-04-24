# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import radarpkg2.processing.symplectic as sym

def K(x):

    return 3 / np.pi * (1-x**2)**2 * (x<1) * (x>-1)

def g(x):
    """ g = - K' """
    return -3 / np.pi * (-4*x) * (1-x**2) * (x<1) * (x>-1)    

def kde(point, sample, radius):

    k = len(sample)
    r = radius
    r = k**(-1/5)

    S = 0
    
    for xi in sample:

        t = sym.theta(xi,point)
        d = sym.dist(point,xi)

        S += 1/r * 1/t * K(d/r)

    return S/k

def kde2(point,sample,cn = 1):

    """ sans theta_0 """

    k = len(sample)
    r = k**(-1/5)

    S = 0
    for xi in sample:

        d = sym.dist(point,xi)

        S += 1/(r**3) * g(d**2 / r**2)

    return cn * S/k

def kde3(point,sample,cn = 1):

    """ avec log """

    k = len(sample)
    r = k**(-1/5)

    S = 0
    for xi in sample:

        d = sym.dist(point,xi)

        S += 1/(r**3) * g(d**2 / r**2) * Rlog(point,xi)

    return cn * S/k

def Rlog(x,xi):
    return x * np.log(1/x * xi)

def mean_shift(sample,tol = 1e-3):
    
    new_sample = []
    
    y = 0j
    
    for xi in sample:
        
        y = xi + 1e-3
        
        res = 10
        
        while (np.absolute(res) > 1):
            res = kde3(y,sample)/kde2(y,sample) * (np.absolute(y) < 1)
            y = y * np.exp(1/y * res)
            print(np.absolute(res))
            
        new_sample.append(y)
        
    return new_sample

def compute_density_on_disk(sample, nx, radius):

    X = np.linspace(-1,1,nx)
    Y = np.linspace(-1,1,nx)
    X,Y = np.meshgrid(X,Y)
    pts = X + 1j*Y

    print(pts.shape)

    density = kde(pts, sample, radius)
    density[np.isnan(density)] = 0

    return density

def compute_mus_density(burg_output, nx, radius, progress=False):

    """ estimation des densités des coefficients de réflexion sur le disque de Poincaré """        

    order = burg_output.shape[-1]
    data = []
    f = lambda x:x
    if progress: f=tqdm
    
    for m in f(range(order)):
        coeffs = burg_output[..., m]
        D = compute_density_on_disk(coeffs, nx, radius)
        data.append(D)

    return np.stack(data)