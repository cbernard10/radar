from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.special import gamma
from tqdm import tqdm
from scipy import stats
from ..visualisation.visualisation import green, red, blue, yellow

from .dataset_to_cdata import dataset_to_cdata
from .dataset_to_burg import dataset_to_burg_jit
from scipy.stats import t

from .geo_student_3d_rk4 import find_geodesic
# from geo_student_3d_rkf6_III import find_geodesic

EPS = 1e-4

def fit_dist(dist, data, rng):
    res = None
    if isinstance(rng, int):
        res = dist.fit(data, loc=0)
    elif rng == 'all':
        # res = dist.fit(np.reshape(data, (data.shape[0] * data.shape[1], -1)), loc=0)
        res = dist.fit(data.flatten())
    # x = np.linspace(np.min(data), np.max(data), 100)

    arg = res[:-2]
    loc = res[-2]
    scale = res[-1]

    # x = np.linspace(np.min(data), np.max(data), 1000)
    # y = dist.pdf(x, *arg, loc, scale) if arg else dist.pdf(x, loc, scale)
    # plt.hist(data.flatten(), bins='auto', density=1)
    # plt.plot(x, y, label=dist.name)
    # plt.legend(loc='upper right')
    # plt.show()

    return (loc, scale, *arg)

def fit_t(data, rng):
    return fit_dist(t, data, rng)

from tqdm import tqdm

def fit_cdata(cdata, rng='all'):

    t_params = np.zeros((cdata.shape[0], cdata.shape[1], 3))

    for i in tqdm(range(cdata.shape[0])):
        for j in range(cdata.shape[1]):

            res = fit_t( np.real(cdata[i, j]), rng=rng )
            t_params[i,j] = res

    return t_params

def dataset_to_student_params(dataset):

    cdata = dataset_to_cdata(dataset, subsampling=1)
    return cdata_to_student_params(cdata)

def cdata_to_student_params(cdata):

    re = np.real(cdata)
    im = np.imag(cdata)

    dist = t
    res = []

    for data in [re, im]:
        params = dist.fit(data.flatten()[::1])
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        res.append([loc, scale, *arg])

    return {'Real': res[0], 'Imag': res[1]}

def save_student_params_for_datasets(save=0):

    from dataset_to_cdata import get_datasets
    from tqdm import tqdm
    from radarpkg.visualisation.visualisation import red, green

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dss = get_datasets()
    params_re = []
    params_im = []
    type = []
    colors = []
    print(dss)
    for ds in tqdm(dss):

        try:
            print(green(ds))
            res = dataset_to_student_params(ds)
            params_re.append(res[0])
            params_im.append(res[1])
            type = ds[7]

            if type=='C':
                colors.append('blue')
            elif type=='T':
                colors.append('green')
            elif type=='N':
                colors.append('green')

            print(f"found params {red(str(params_re[-1]))},  {red(str(params_im[-1]))}")

        except Exception as e:
            print(red(str(e)))
            continue

    if(save):
        np.save('student_params_re.npy', params_re)
        np.save('student_params_im.npy', params_im)

    return params_re, params_im, colors

# def kld_params_(ps1, ps2):
#     x = np.linspace(-30, 30, 1000)
#     p1 = t.pdf(x, ps1[2], loc=ps1[0], scale=ps1[1])
#     p2 = t.pdf(x, ps2[2], loc=ps2[0], scale=ps2[1])
#
#     return np.sum(rel_entr(p1, p2))

from scipy.integrate import quad
def kld_params(ps1, ps2):
    # x = np.linspace(-30, 30, 1000)
    p1 = lambda x: t.pdf(x, ps1[2], loc=ps1[0], scale=ps1[1])
    p2 = lambda x: t.pdf(x, ps2[2], loc=ps2[0], scale=ps2[1])

    res = quad(lambda x: p1(x) * np.log(p1(x) / p2(x)), a=-np.inf, b=np.inf)
    # print(res)
    return res[0]

    # return np.sum(rel_entr(p1, p2))

def frechet_var(X, p, pow=pow, weights = []):

    # print(np.array(X).shape)
    if weights == []:
        weights = np.ones((np.array(X).shape[0]))

    x = np.linspace(-25, 25, 30000)
    # p1 = t.pdf(x, p[2], loc=p[0], scale=p[1])

    su = 0

    for i, xi in enumerate(X):
        # p2 = t.pdf(x, xi[2], loc=xi[0], scale=xi[1])
        dist_sq = weights[i] * kld_params(p, xi)**pow
        su += dist_sq

    # print(su)
    return su

def karcher_mean(X, pow=2, weights = []):

    if weights == []:
        weights = np.ones((np.array(X).shape[0]))

    from scipy.optimize import minimize
    x0 = np.mean(X, axis=0)
    res = minimize(lambda p:frechet_var(X, p, pow=pow, weights=weights), x0, method = 'Nelder-Mead')
    return res['x']

def _loss_geodesic(start, end, dX, dt, verbose=False):
    dx, dy, dz = dX
    X, Y, Z = find_geodesic(start[0], dx, start[1], dy, start[2], dz, 1, dt=dt)

    estimated_end = [X[-1], Y[-1], Z[-1]]
    d = np.sqrt((end[0] - X[-1])**2 + (end[1] - Y[-1])**2 + (end[2] - Z[-1])**2)
    if verbose: print(yellow(dX), blue(estimated_end), green(end), red(d))
    return d

# def _loss_geodesic_speed_for_radius(start, dX0, a, radius, dt):
#
#     # find the variable a such that dX = a * dX0 makes a geodesic of length 1 at T=1.
#
#     a=1.0
#     dx, dy, dz = a * dX0
#     X, Y, Z = find_geodesic(start[0], dx, start[1], dy, start[2], dz, 1, dt=dt)
#
#     estimated_length = geodesic_length(X, Y, Z)
#
#     loss = np.abs(radius - estimated_length)**2
#
#     print(yellow(dX0), blue(estimated_length), green(end), red(d))
#
#     return loss

def get_initial_conditions_from_end_pt(start, end, xatol, fatol, dt, verbose=False):

    from scipy.optimize import minimize
    dX0 = np.array([end[0]-start[0], end[1]-start[1], end[2]-start[2]])/10
    res = minimize(lambda p: _loss_geodesic(start, end, p, dt, verbose=verbose), np.array([*dX0]), method='Nelder-Mead',
                   options={'xatol': xatol, 'fatol': fatol})

    return res['x']

# def get_v0_for_radius(start, radius, dX0, xatol, fatol, dt):
#
#     from scipy.optimize import minimize
#     a=1
#     res = minimize(lambda p: _loss_geodesic_speed_for_radius(start, dX0, a, radius, dt), np.array([a]), method='Nelder-Mead',
#                    options={'xatol': xatol, 'fatol': fatol})
#
#     return res['x']

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

def plot_datasets():
    from dataset_to_cdata import get_datasets

    colors = []

    for d in get_datasets():
        type = d[7]
        if type == 'C':
            colors.append('#0000ff')
        elif type == 'N':
            colors.append('#ffa500')
        elif type == 'T':
            colors.append('#00ff00')

    re = np.load('student_params_im.npy')

    args = np.argwhere(re[..., -1] > 0.4)
    re = np.squeeze(re[args])
    colors = np.squeeze(np.array(colors)[args])
    return plot_pts_and_mean(re, colors)

def student_cylinder(X):

    x,y,z = X

    return [2*x / (x**2 + (1+y)**2), (x**2 + y**2 - 1)/(x**2 + (1+y)**2), z]

def geodesic_length(X, Y, Z):

    sum = 0

    for i in range(len(X)-1):

        sum += np.sqrt( (X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2 )

    return sum

def geodesic_between_start_end_pts(start, end, xatol=1e-6, fatol=1e-6, dt=1e-3, verbose=False):

    if verbose: print(f"computing {blue(start)} -> {blue(end)}")
    dX = get_initial_conditions_from_end_pt(start, end, xatol, fatol, dt, verbose=verbose)
    X, Y, Z = find_geodesic(start[0], dX[0], start[1], dX[1], start[2], dX[2], 1, dt)

    dX_ = [-X[-1] + X[-2], -Y[-1] + Y[-2], -Z[-1] + Z[-2]]

    kld = kld_params(start, end)
    geo_len = geodesic_length(X, Y, Z)

    return X, Y, Z, dX, dX_, kld, geo_len

# def geodesic_ball(center, radius):
#
#     v0s = []
#
#     for _ in range(100):
#
#         theta, phi = np.random.rand()*np.pi, np.random.rand()*2*np.pi
#         [x, y, z] = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
#         v0s.append([x, y, z])
#
#     for v0 in v0s:
#
#         X, Y, Z =
#
#     return 0


def plot_geodesic_between_start_end_pts(start, end, xatol, fatol, dt):
    dX = get_initial_conditions_from_end_pt(start, end, xatol, fatol, dt)

    print(dX)
    X, Y, Z = find_geodesic(start[0], dX[0], start[1], dX[1], start[2], dX[2], 1, dt)

    plot_geodesic(X, Y, Z)


""" KLD entre images par range distance proche moyenne et grande (p11 p21 p31), (p12 p22 p32) """
""" dist somme des KLD range par range """

def draw_geodesic_triangle(p1, p2, p3):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_zlabel('ν')

    vectors = []
    angles = []

    for pair in [(p1, p2), (p2, p3), (p3, p1)]:

        x, y, z, dx, dx_ = geodesic_between_start_end_pts(pair[0], pair[1])
        vectors.append((dx, dx_))
        ax.plot(x, y, z, color='m')

    v0, v1, v2, v3, v4, v5 = vectors[0][1][1:], vectors[1][0][1:], vectors[1][1][1:], vectors[2][0][1:], vectors[2][1][1:], vectors[0][0][1:]

    x_pos = [p1[0], p2[0], p3[0]]
    y_pos = [p1[1], p2[1], p3[1]]
    z_pos = [p1[2], p2[2], p3[2]]

    u_dir = [vectors[0][0][0], vectors[1][0][0], vectors[2][0][0]]
    v_dir = [vectors[0][0][1], vectors[1][0][1], vectors[2][0][1]]
    w_dir = [vectors[0][0][2], vectors[1][0][2], vectors[2][0][2]]

    ax.quiver(x_pos, y_pos, z_pos, u_dir, v_dir, w_dir, length=0.5, normalize=True, color='k')

    x_pos = [p2[0], p3[0], p1[0]]
    y_pos = [p2[1], p3[1], p1[1]]
    z_pos = [p2[2], p3[2], p1[2]]

    u_dir = [vectors[0][1][0], vectors[1][1][0], vectors[2][1][0]]
    v_dir = [vectors[0][1][1], vectors[1][1][1], vectors[2][1][1]]
    w_dir = [vectors[0][1][2], vectors[1][1][2], vectors[2][1][2]]

    ax.quiver(x_pos, y_pos, z_pos, u_dir, v_dir, w_dir, length=0.5, normalize=True, color='k')

    angles.append(np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))))
    angles.append(np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))))
    angles.append(np.arccos(np.dot(v4, v5) / (np.linalg.norm(v4) * np.linalg.norm(v5))))
    # angles.append(np.arctan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))
    # angles.append(np.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
    # angles.append(np.arctan2(np.linalg.det([v2, v3]), np.dot(v2, v3)))
    # angles.append(np.atan2(np.linalg.det([v3, v4]), np.dot(v3, v4)))
    # angles.append(np.arctan2(np.linalg.det([v4, v5]), np.dot(v4, v5)))
    # angles.append(np.atan2(np.linalg.det([v5, v0]), np.dot(v5, v0)))

    print(f"somme des angles {np.degrees(sum(angles))}")
    fig.tight_layout()

def cdata_ranges_to_student_params(cdata):
    n_range_bins = cdata.shape[1]
    params_array = []
    for i in tqdm(range(n_range_bins)):

        range_data = cdata[:, i, :]
        re = np.real(range_data)
        im = np.imag(range_data)

        dist = t

        for data in [re]:
            params = dist.fit(data.flatten()[::1])
            # x = np.linspace(np.min(data), np.max(data), 100)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            params_array.append([loc, scale, *arg])

    return np.array(params_array)

def dataset_ranges_to_student_params(dataset):
    cdata = dataset_to_cdata(dataset, subsampling=1)
    return cdata_ranges_to_student_params(cdata)

def plot_pts_and_mean(X, colors='b', mean=False, paths=False, labels=False, with_labels=False, s=10):

    from mpl_toolkits.mplot3d import Axes3D

    if mean:
        mean = karcher_mean(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = X[..., 0], X[..., 1], X[..., 2]

    # Z = np.log(Z)
    #
    # print(X, Y, Z, colors)

    ax.scatter(X, Y, Z, color=colors, s=s)
    if paths:
        ax.plot(X, Y, Z, c='gray')
    if mean:
        ax.scatter([mean[0]], [mean[1]], [mean[2]], c='red', s=s)
    if labels:
        for i in range(X.shape[0]):
            ax.text(*(X[i], Y[i], Z[i]), f"{i}")

    ax.set_zlim3d(0, 20)
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_zlabel('ν')
    plt.show()
    return ax

"02_021_NStFA cible avec boucle"
"04_022_TTrFA cible (49-75) transition entre 44 et 48 puis entre 76 et 100"
"06_004_TTrFA cible 54"
"06_015_TTrFA cible : degrés de liberté se rapprochent de 0"
"06_017_TTrFA cible : écart-type augmente "
"06_048_TTrFA cible: nu entre 5 et 15 "
"06_050_TTrFA cibles en bas"
"06_053_TTrFA cible (24-28) : changement brutal des degrés de liberté contrairement au ground clutter (0 -> 10)"
"08_028_TStFA cible claire"
"08_033_NTrFA"
"08_046_NTrFA"
"09_071_NTrFA boucle"
"11_019_CStFA tas uniforme -> clutter"

""" relier les différents points par des géodésiques au lieu de lignes droites """

# if __name__ == '__main__':

# prms = ranges_to_student_params(' 04_022_TTrFA')
#
# print(prms)
# plot_pts_and_mean(prms, 'blue', True)

    # X, Y = np.linspace(-2, 2, 20), np.linspace(0, 0.76, 100)
    # XX, YY = np.meshgrid(X, Y)
    #
    # Z = 1/(5*(-YY + 0.76))
    #
    # ax = plot_datasets()
    # ax.plot_surface(XX, YY, Z, color='r', alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # draw_geodesic_triangle((1, 1, 1), (1, 1, 2), (1, 2, 2))
    # draw_geodesic_triangle((1, 1, 1), (1, 1, 4), (1, 4, 4))
    # draw_geodesic_triangle((1, 1, 1), (1, 1, 6), (1, 6, 6))
    # draw_geodesic_triangle((1, 1, 1), (1, 1, 8), (1, 8, 8))
    #
    # plt.show()