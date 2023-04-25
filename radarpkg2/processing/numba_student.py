import numpy as np
import matplotlib.pyplot as plt
from ..visualisation.visualisation import green, red, blue, yellow
from .geo_student_3d_rk4 import find_geodesic

from scipy.integrate import quad
from scipy.stats import t

import numba

EPS = 1e-4

def kld_params(ps1, ps2):
    # x = np.linspace(-30, 30, 1000)
    p1 = lambda x: t.pdf(x, ps1[2], loc=ps1[0], scale=ps1[1])
    p2 = lambda x: t.pdf(x, ps2[2], loc=ps2[0], scale=ps2[1])

    res = quad(lambda x: p1(x) * np.log(p1(x) / p2(x)), a=-np.inf, b=np.inf)
    # print(res)
    return res[0]

def _loss_geodesic(start, end, dX, dt, verbose=False):
    dx, dy, dz = dX
    X, Y, Z = find_geodesic(start[0], dx, start[1], dy, start[2], dz, 1, dt=dt)

    estimated_end = [X[-1], Y[-1], Z[-1]]
    d = min((end[0] - X[-1])**2 + (end[1] - Y[-1])**2 + (end[2] - Z[-1])**2 + 1 * (end[0] - X[-1])**2, 10000)
    if np.isnan(d): return 10000
    d = d
    if verbose: print(yellow(dX), blue(estimated_end), start, green(end), red(d))
    return d

def get_initial_conditions_from_end_pt(start, end, xatol, fatol, dt, verbose=False):

    from scipy.optimize import minimize, basinhopping, dual_annealing, newton, root
    dX0 = np.array([end[0]-start[0], end[1]-start[1], end[2]-start[2]])/10

    res = minimize(lambda p: _loss_geodesic(start, end, p, dt, verbose=verbose), np.array([*dX0]), method='Nelder-Mead', bounds=((-5, 5), (-5, 5), (-20, 20)),
                   options={'xatol': xatol, 'fatol': fatol})

    # def a(f_new, x_new, f_old, x_old):
    #
    # res = basinhopping(
    #     lambda p: _loss_geodesic(start, end, p, dt, verbose=verbose),
    #     np.array([*dX0]),
    #     stepsize=0.5, T=1,
    # accept_test=a )

    # res = dual_annealing(
    #     lambda p: _loss_geodesic(start, end, p, dt, verbose=verbose),
    #     bounds=[ (-10, 10), (-10, 10), (-10, 10) ]
    # )

    # res = root(
    #     lambda p: _loss_geodesic(start, end, p, dt, verbose=verbose),
    #     np.array([*dX0])
    # )

    return res['x']

def plot_geodesic(X, Y, Z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim3d(-0.5, 0.5)
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_zlabel('ν')
    ax.plot(X, Y, Z)
    plt.tight_layout()
    plt.show()

def geodesic_length(X, Y, Z):

    sum = 0.0
    for i in range(len(X)-1):
        sum += np.sqrt( (X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2 )
    return sum


def geodesic_between_start_end_pts(start, end, xatol=1e-3, fatol=1e-3, dt=5e-2, verbose=0):

    if verbose: print(f"computing {blue(start)} -> {blue(end)}")

    dX = get_initial_conditions_from_end_pt(start, end, xatol, fatol, dt, verbose=verbose)
    X, Y, Z = find_geodesic(start[0], dX[0], start[1], dX[1], start[2], dX[2], 1, dt)
    dX_ = [-X[-1] + X[-2], -Y[-1] + Y[-2], -Z[-1] + Z[-2]]

    # kld = kld_params(start, end)
    kld = 0

    geo_len = geodesic_length(X, Y, Z)

    return X, Y, Z, dX, dX_, kld, geo_len

def dist_between_pts(pair_of_pts):
    if(list(pair_of_pts[0]) == list(pair_of_pts[1])):
        return 0
    else:
        return geodesic_between_start_end_pts(pair_of_pts[0], pair_of_pts[1])[-1]

from tqdm import tqdm
import os

def distance_matrix(pts, verbose=0, filename='./distance_matrix_student', autosave=5):

    batch_size = len(pts)
    dist_mat = np.zeros((batch_size, batch_size))

    for i in tqdm(range(batch_size)):
        for j in tqdm(range(i, batch_size)):

            if i==j:
                continue

            start, end = pts[i], pts[j]

            start = (start[0], start[1], min(30.0, start[2]))
            end = (end[0], end[1], min(30.0, end[2]))

            dist_mat[i,j] = geodesic_between_start_end_pts(start, end, verbose=verbose)[-1]

            if i%autosave == 0:
                np.save(filename, dist_mat)

    return dist_mat + dist_mat.T

from multiprocessing import Pool

def distance_matrix_pooled(pts, verbose=0, filename='./distance_matrix_student', autosave=5):

    n_pts = len(pts)

    pairs = []

    for i in tqdm(range(n_pts)):
        for j in tqdm(range(i+1, n_pts)):

            pairs.append((pts[i], pts[j]))

    n_pairs = len(pairs)

    print(n_pairs)
    print(n_pts)
    assert(n_pairs == n_pts * (n_pts-1) / 2)

    with Pool() as p:
        dist_mat = list(tqdm(p.imap(dist_between_pts, pairs), total=n_pairs))

    def create_upper_matrix(values, size):
        upper = np.zeros((size, size))
        upper[np.triu_indices(n_pts, 1)] = values
        return (upper)

    dist_mat = create_upper_matrix(dist_mat, n_pts)

    np.save(filename, dist_mat.T + dist_mat)

    return dist_mat.T + dist_mat

