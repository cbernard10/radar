# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import polygamma
from mpl_toolkits.mplot3d import Axes3D
import time
from math import factorial

def polygamma_stirling(n, x):

    # if x < 32:
    #     return polygamma(n, x)
    #
    # else:
    return polygamma(n, x)

    #%% polygamma(n, x+1) = polygamma_stirling(n, x)... putain de merde
    # x+=1
    # if(n == 1):
    #     return -1/(2*x**2) + 1/x + 1/(6*(x+1)**3) - 1/12*( 1/((x+1)**4 * (x+2)**4) *
    #                                                           (2*((x+1)**2 * (x+2)**2) - (2*x-3) * (2*(x+1)*((x+2)**2) + 2*((x+1)**2 * (x+2)))))

    # if(n == 1):
    #     return -0.5/(x**2) + 1/(6*(x+1)**3 * (x+2)) + 1/(6*(x+1)**2 * (x+2)**2) + 1/(6*(x+1)**3) + 1/(6*(x+1) * (x+2)**3) + 1/x
    #
    # elif(n == 2):
    #     return -1/(x**3) - 1/(x**2)

    if (n == 1):
        return 1/x -0.5 / (x ** 2) + 1 / (6 * (x + 1) ** 3)

    elif (n == 2):
        return 1 / (x ** 3) - 1 / (x ** 2) - 0.5/((x+1)**4)

    else:
        return (-1)**n * (factorial(n-2) / (x**(n-1)) - factorial(n-1) / (2*x**n) )

def g11(v, sigma):
    return (v + 1) / (v + 3) * sigma ** (-2)


def g22(v, sigma):
    return v / (2 * (v + 3)) * sigma ** (-4)


def g33(v, sigma):
    return -0.5 * (
                0.5 * (polygamma_stirling(1, (v + 1) / 2) - polygamma_stirling(1, v / 2)) + 1 / (v * (v + 1)) - 1 / (v + 1) + (v + 2) / (
                    v * (v + 3)))

def g23(v, sigma):
    return -1 / ((v + 3) * (v + 1) * sigma ** 2)


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

# def inverse_metric_(v, sigma):
#     inv =  np.linalg.inv(metric(v, sigma))
#     # print(inv)
#     return inv

def inverse_metric(v, sigma):
    det = g11(v, sigma) * (g33(v, sigma) * g22(v, sigma) - g23(v, sigma) * g32(v, sigma))

    return np.array([
        [((v+3) * sigma**2)/(v+1), 0, 0],
        [0, g11(v, sigma)*g33(v, sigma) / (det+1e-5), -g11(v, sigma)*g23(v, sigma) / (det+1e-5)],
        [0, -g11(v, sigma)*g23(v, sigma) / (det+1e-5), g11(v, sigma)*g22(v, sigma) / (det+1e-5)]
    ])

#%%

def dg(a, b, c, v, sigma):
    if c == 1:
        if (a == 0 and b == 0):
            return -2 / (sigma ** 3) * (v + 1) / (v + 3)

        elif (a == 1 and b == 1):
            return -2 / (sigma ** 5) * v / (v + 3)

        elif (a == 1 and b == 2 or a == 2 and b == 1):
            return 2 / (sigma ** 3) * 1/((v+1)*(v+3))

        else:
            return 0

    elif c == 2:
        if (a == 0 and b == 0):
            return 2 / (sigma ** 2 * (v + 3) ** 2)

        elif (a == 1 and b == 1):
            return 1.5 / ((v + 3) ** 2 * sigma ** 4)

        elif (a == 2 and b == 2):
            return -0.5 * (0.25 * polygamma_stirling(2, (v + 1) / 2) - 0.25 * polygamma_stirling(2, v / 2) - (2 * v + 1) / (
                        v ** 2 * (v + 1) ** 2) + 1 / ((v + 1) ** 2) + (-v ** 2 - 4 * v - 6) / (v ** 2 * (v + 3) ** 2))

        elif (a == 2 and b == 1 or a == 1 and b == 2):
            return (2 * v + 4) / ((sigma ** 2 * (v + 3) ** 2 * (v + 1) ** 2))

        else:
            return 0

    else:
        return 0

    # dG = np.zeros((3, 3, 3))
    # dG[0, 0, 1] = -2/(sigma**3) * (v+1)/(v+3)
    # dG[1, 1, 1] = -2/(sigma**5) * v/(v+3)
    # dG[1, 2, 1] = 2/(sigma**3) * 1/((v+1)*(v+3))
    # dG[2, 1, 1] = dG[1, 2, 1]
    #
    # dG[0, 0, 2] = 2/(sigma**2 * (v+3)**2 )
    # dG[1, 1, 2] = 1.5 / ((v+3)**2 * sigma**4)
    # dG[2, 2, 2] = -0.5 * ( 0.25 * polygamma(2, (v+1)/2) - 0.25 * polygamma(2, v/2) - (2*v+1)/(v**2 * (v+1)**2) - 1/((v+1)**2) - (v**2 - 8*v - 3)/(v**2 * (v+3)**2) )
    # dG[1, 2, 2] = (2*v + 4) / ((sigma**4 * (v+3)**2 * (v+1) ** 2))
    # dG[2, 1, 2] = dG[1, 2, 2]


def christ(i, j, k, v, sigma):
    GI = inverse_metric(v, sigma)

    sum = 0

    for l in range(3):
        sum += 0.5 * GI[i, l] * (dg(j, l, k, v, sigma) + dg(k, l, j, v, sigma) - dg(j, k, l, v, sigma))

    return sum
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(3):
    #             for l in range(3):
    #
    #                 C[i, j, k] = 0.5 * GI[i, l] * (dg(j, l, k) + dg(k, l, j) - dg(j, k, l))


def find_geodesic(X0, X_0, Y0, Y_0, Z0, Z_0, Tmax, dt, idx=-1):
    T = np.arange(0, Tmax, dt)

    X, X_, Y, Y_, Z, Z_ = [X0],[X_0],[Y0],[Y_0], [Z0], [Z_0]
    Xnext, X_next, Ynext, Y_next, Znext, Z_next = X.copy(), X_.copy(), Y.copy(), Y_.copy(), Z.copy(), Z_.copy()

    """ équation des géodésiques, Euler explicite """
    """ X_ = d/dt X, Y_ = d/dt Y """
    """ mu et nu """

    for i in range(len(T) - 1):
        def steps(x, x_, y, y_, z, z_, dt):

            def fx(x, x_, y, y_, z, z_, dt):
                return -2 * (christ(0, 0, 1, z, y) * x_ * y_ + christ(0, 0, 2, z, y) * x_ * z_)

            def fy(x, x_, y, y_, z, z_, dt):
                return -2 * christ(1, 2, 1, z, y) * y_ * z_ - christ(1, 0, 0, z, y) * x_ ** 2 - christ(1, 1, 1, z,
                                                                                                      y) * y_ ** 2 - christ(
                    1, 2, 2, z, y) * z_ ** 2

            def fz(x, x_, y, y_, z, z_, dt):
                return -2 * christ(2, 1, 2, z, y) * y_ * z_ - christ(2, 0, 0, z, y) * x_ ** 2 - christ(2, 1, 1, z,
                                                                                                      y) * y_ ** 2 - christ(
                    2, 2, 2, z, y) * z_ ** 2

            k1 = [
                dt*fx(x, x_, y, y_, z, z_, dt),
                dt*fy(x, x_, y, y_, z, z_, dt),
                dt*fz(x, x_, y, y_, z, z_, dt)
            ]
            k2 = [
                dt*fx(x, x_+0.5*k1[0], y, y_+0.5*k1[1], z, z_+0.5*k1[2], dt),
                dt*fy(x, x_+0.5*k1[0], y, y_+0.5*k1[1], z, z_+0.5*k1[2], dt),
                dt*fz(x, x_+0.5*k1[0], y, y_+0.5*k1[1], z, z_+0.5*k1[2], dt)
            ]
            k3 = [
                dt*fx(x, x_ + 0.5 * k2[0], y, y_ + 0.5 * k2[1], z, z_ + 0.5 * k2[2], dt),
                dt*fy(x, x_ + 0.5 * k2[0], y, y_ + 0.5 * k2[1], z, z_ + 0.5 * k2[2], dt),
                dt*fz(x, x_ + 0.5 * k2[0], y, y_ + 0.5 * k2[1], z, z_ + 0.5 * k2[2], dt)
            ]
            k4 = [
                dt*fx(x, x_ + k3[0], y, y_ + k3[1], z, z_ + k3[2], dt),
                dt*fy(x, x_ + k3[0], y, y_ + k3[1], z, z_ + k3[2], dt),
                dt*fz(x, x_ + k3[0], y, y_ + k3[1], z, z_ + k3[2], dt)
            ]
            s = 1/6 * np.array([
                k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0],
                k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1],
                k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2],
            ])

            return s

        s = steps(X[-1], X_[-1], Y[-1], Y_[-1], Z[-1], Z_[-1], dt)

        X_next.append(X_[-1] + s[0])
        Xnext.append(X[-1] + dt * X_next[-1])
        Y_next.append(Y_[-1] + s[1])
        Ynext.append(Y[-1] + dt * Y_next[-1])
        Z_next.append(Z_[-1] + s[2])
        Znext.append(Z[-1] + dt * Z_next[-1])

        X, X_, Y, Y_, Z, Z_ = Xnext, X_next, Ynext, Y_next, Znext, Z_next

    return X, Y, Z


# nu = 1
# sigma = 1
#
# print(inverse_metric(nu, sigma))
#
# ch = np.zeros((3, 3, 3))
#
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#
#             ch[i, j, k] = christ(i, j, k, nu, sigma)
#
# print(ch[2])

# %%

if __name__=='__main__':
    Tmax = 1

    dt = 1e-3
    x0 = 0
    dx = 1
    y0 = 1
    dy = 0
    z0 = 6
    dz = 0

    paths = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for z0 in tqdm(np.linspace(16, 256, 5)):
        for y0 in tqdm(np.linspace(1.5, 5, 5)):

            x0 = 0

            dx, dy, dz = 1, 0, 0

            X, Y, Z = find_geodesic(x0, dx, y0, dy, z0, dz, Tmax, dt, idx=f"{[y0, z0]}")
            paths.append([X, Y, Z])
            ax.plot(X, Y, Z)

    #%%
    # for z0 in tqdm(np.linspace(16, 256, 5)):
    #     for y0 in tqdm(np.linspace(1.5, 5, 5)):
    #
    #         x0 = 0
    #         dx, dy, dz = 1, 1, 1
    #
    #         X, Y, Z = find_geodesic(x0, dx, y0, dy, z0, dz, Tmax, dt, idx=f"{[y0, z0]}")
    #         paths.append([X, Y, Z])
    #         ax.plot(X, Y, Z)

    #%%
    # for x0 in tqdm(np.linspace(0, 2, 5)):
    #     for y0 in tqdm(np.linspace(1.5, 5, 5)):
    #
    #         z0 = 3
    #         dx, dy, dz = 0, 0, 0.5
    #
    #         X, Y, Z = find_geodesic(x0, dx, y0, dy, z0, dz, Tmax, dt, idx=f"{[y0, z0]}")
    #         paths.append([X, Y, Z])
    #         ax.plot(X, Y, Z)

    #%%
    # for x0 in tqdm(np.linspace(0, 2, 5)):
    #     for z0 in tqdm(np.linspace(16, 64, 5)):
    #
    #         y0 = 1
    #         dx, dy, dz = 0, 1, 0
    #
    #         X, Y, Z = find_geodesic(x0, dx, y0, dy, z0, dz, Tmax, dt, idx=f"{[y0, z0]}")
    #         paths.append([X, Y, Z])
    #         ax.plot(X, Y, Z)

    #%%
    mumax = -np.inf
    sigmamax = 0
    numax = 0

    for path in paths:

        mu, sigma, nu = path[0], path[1], path[2]

        if max(mu) > mumax: mumax = max(mu)
        if max(sigma) > sigmamax: sigmamax = max(sigma)
        if max(nu) > numax: numax = max(nu)
        print(min(nu))

    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_zlabel('ν')

    ax.set_xlim3d(0, mumax*1.1)
    ax.set_ylim3d(0, sigmamax*1.1)
    ax.set_zlim3d(0, numax*1.1)

    plt.tight_layout()

    # for angle in range(0, 360):
    #     if angle==359:
    #         angle=0
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    plt.show()

    # plt.grid(1)
    # plt.yscale('log')
    # plt.show()