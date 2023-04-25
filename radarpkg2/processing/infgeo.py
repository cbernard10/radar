from scipy import stats
import numpy as np
from radarpkg2.processing.dataset_to_cdata import dataset_to_cdata
from radarpkg2.processing.dataset_to_burg import dataset_to_burg_jit
import matplotlib.pyplot as plt
from scipy.special import gamma, rel_entr
from tqdm import tqdm

from radarpkg2.visualisation.visualisation import col

def get_post_dist_list():

    return [
        'gamma', 'moyal', 'mielke', 'invweibull', 'chi2', 'levy', 'fisk', 'genextreme', 'fatiguelife', 'lognorm', 'k'
    ]

    # return [
    #     stats.gamma, stats.moyal, stats.mielke, stats.invweibull, stats.chi2, stats.levy, stats.fisk, stats.genextreme, stats.fatiguelife, stats.lognorm
    # ]

def get_symm_dists():

    return [
        't', 'norm', 'hypsecant', 'logistic', 'cauchy'
    ]

    # return [
    #     stats.t, stats.norm, stats.genhyperbolic, stats.hypsecant, stats.logistic, stats.cauchy
    # ]

def intensity(cdata):
    return np.linalg.norm(cdata, axis=-1)


def dataset_to_range_hist(dataset, rng='all'):
    cdata = dataset_to_cdata(dataset)
    return cdata_to_range_hist(cdata, rng)

def cdata_to_range_hist(cdata, rng='all'):
    if rng == 'all':
        cdata_range = intensity(np.reshape(cdata, (cdata.shape[0] * cdata.shape[1], -1)))
    elif isinstance(rng, int):
        cdata_range = intensity(cdata[:, rng])
    else:
        return None
    # plt.hist(cdata_range, bins='auto', rwidth=0.5, density=True)
    return np.median(cdata_range), np.mean(cdata_range), np.std(cdata_range)


def find_moments_variation_for_dataset(dataset, ss=1):
    cdata = dataset_to_cdata(dataset)[:, ::ss]
    ranges = cdata.shape[1]

    pw = intensity(cdata)
    moments = {
        'median': [],
        'mean': [],
        'std': [],
        'dispmedian': 0,
        'dispmean': 0,
        'kld_to_global': [],
        'dispkld': 0
    }

    for i in range(ranges):

        pw_range = pw[:, i]
        moments['median'].append(np.median(pw_range))
        moments['mean'].append(np.mean(pw_range))
        moments['std'].append(np.std(pw_range))
        try:
            y1 = intensity_to_kde(pw_range)
            y2 = intensity_to_kde(pw.flatten())
            y1, y2 = y1[np.logical_and(y1 != 0, y2 != 0)], y2[np.logical_and(y1 != 0, y2 != 0)]
            moments['kld_to_global'].append(kld(y1, y2))
        except np.linalg.LinAlgError as _:
            moments['kld_to_global'].append(10)

    def quartile_coefficient_of_dispersion(data):
        return (np.quantile(data, 0.75) - np.quantile(data, 0.25)) / (np.quantile(data, 0.75) + np.quantile(data, 0.25))

    moments['dispmedian'] = quartile_coefficient_of_dispersion(moments['median'])
    moments['dispmean'] = quartile_coefficient_of_dispersion(moments['mean'])
    moments['dispkld'] = quartile_coefficient_of_dispersion(moments['kld_to_global'])

    return moments


def dataset_range_hist_fit(dataset, dist, rng='all'):
    cdata = dataset_to_cdata(dataset)
    return cdata_range_hist_fit(cdata, dist, rng)

def cdata_range_hist_fit(cdata, dist, rng='all'):
    res = None
    if isinstance(rng, int):
        res = dist.fit(intensity(cdata[rng]), loc=0)
    elif rng == 'all':
        res = dist.fit(intensity(np.reshape(cdata, (cdata.shape[0] * cdata.shape[1], -1))), loc=0)
    return res

def dataset_range_hist_fit_distributions(dataset, dist_list, rng='all', subsampling = 1):
    cdata = dataset_to_cdata(dataset, subsampling)
    return cdata_range_hist_fit_distributions(cdata, dist_list, rng)

def cdata_range_hist_fit_distributions(cdata, dist_list, rng='all'):
    params = []
    klds = []

    for dist in dist_list:
        print(f"fitting {col(dist.name, 'red')}")
        res = cdata_range_hist_fit(cdata, dist, rng)
        pw = intensity(cdata).flatten()
        x = np.linspace(0, np.max(pw), 100)
        gauss_dist = intensity_to_kde(pw)

        arg = res[:-2]
        loc = res[-2]
        scale = res[-1]

        p2 = dist.pdf(x, *arg, loc, scale) if arg else dist.pdf(x, loc, scale)

        gauss_dist, p2 = gauss_dist[np.logical_and(gauss_dist != 0, p2 != 0)], p2[
            np.logical_and(gauss_dist != 0, p2 != 0)]
        klds.append(kld(gauss_dist, p2))

        params = [*params, (arg, loc, scale)]

    return [dist_list, params, klds]


def show_fitted_distributions_for_dataset_and_range(dataset, dist_list, rng, vmin=0, vmax=50):
    cdata = dataset_to_cdata(dataset)
    show_fitted_distributions_for_cdata_and_range(cdata, dist_list, rng, vmin=0, vmax=50)

def show_fitted_distributions_for_cdata_and_range(cdata, dist_list, rng, vmin=0, vmax=50):
    _, params, _ = cdata_range_hist_fit_distributions(cdata, dist_list, rng)

    for dist, ps in zip(dist_list, params):
        print(ps)
        arg, loc, scale = ps
        # print(f"plotting {dist}")
        x = np.linspace(vmin, vmax, 1000)
        y = dist.pdf(x, *arg, loc, scale) if arg else dist.pdf(x, loc, scale)
        plt.plot(x, y, label=dist)
        plt.legend(loc='upper right')

    cdata_to_range_hist(cdata, rng)
    plt.show()

def kld(p1, p2):
    res = (np.sum(rel_entr(p1, p2)) + np.sum(rel_entr(p2, p1))) / 2
    return res


def asym_kld(p1, p2):
    return np.sum(rel_entr(p1, p2))


def intensity_to_kde(pw):
    return stats.gaussian_kde(pw).pdf(
        np.linspace(0, np.max(pw), 100)
    )


def compute_kld_for_kde_and_fit_distribution(dataset, distribution, rng):
    pw = intensity(dataset_to_cdata(dataset)).flatten()

    p1 = stats.gaussian_kde(pw)
    x = np.linspace(0, np.max(pw), 100)
    y1 = p1.pdf(x)

    params = dataset_range_hist_fit(dataset, distribution, rng)
    y2 = distribution.pdf(x, params[0], loc=params[1], scale=params[2]) if params[0] else distribution.pdf(x,
                                                                                                           loc=params[
                                                                                                               1],
                                                                                                           scale=params[
                                                                                                               2])

    y1, y2 = y1[np.logical_and(y1 != 0, y2 != 0)], y2[np.logical_and(y1 != 0, y2 != 0)]

    return np.sum(kld(y1, y2))


def make_kld_matrix_for_ranges(dataset):
    intensity_map = intensity(dataset_to_cdata(dataset, normalized=False))[::1]

    kld_mat = np.zeros((intensity_map.shape[0], intensity_map.shape[0]))

    for i in tqdm(range(intensity_map.shape[0])):
        for j in range(intensity_map.shape[0]):
            if j < i:
                continue
            else:
                kld_mat[i, j] = kld(
                    intensity_to_kde(
                        intensity_map[i]
                    ),
                    intensity_to_kde(
                        intensity_map[j]
                    )
                )

    return kld_mat + kld_mat.T


def dataset_to_neighbor_distance_map(dataset, reach, accuracy=2):
    accuracy = accuracy if accuracy <= 8 else 8
    # cdata = dataset_to_cdata(dataset)
    cdata = dataset_to_burg_jit(dataset, order=6, gamma=0.05)[..., 1:]
    cdata_padded = cdata[reach: cdata.shape[0] - reach, reach: cdata.shape[1] - reach]
    dist_matrix = np.zeros((cdata.shape[0], cdata.shape[1]), dtype=np.float32)

    def get_neighbors(i, j):

        def get_outer_layer(matrix):

            return [*matrix[0, :].flatten(), *matrix[-1, :].flatten(), *matrix[:, 0].flatten(),
                    *matrix[:, -1].flatten()]

        neighbors = []
        distances = []
        deltas_to_neighbors = []

        def dist(z1, z2):

            r = np.linalg.norm(z2 - z1)
            return r

        for k in range(1, reach + 1):
            x = np.arange(-k, k + 1, 1)
            di, dj = np.meshgrid(x, x)
            delta_matrix = di + 1j * dj
            candidates = get_outer_layer(delta_matrix)
            n = accuracy * k  # number of neighbors to get
            deltas = np.random.choice(candidates, size=n, replace=False)
            for delta in deltas:
                neighbors = [*neighbors, cdata[i + int(np.real(delta)), j + int(np.imag(delta))]]
                distances = [*distances, dist(cdata_padded[i, j], neighbors[-1])]
                deltas_to_neighbors = [*deltas_to_neighbors, delta]

        return neighbors, distances, deltas_to_neighbors

    def avg_distance_matrix():

        for i in tqdm(range(cdata_padded.shape[0])):
            for j in range(cdata_padded.shape[1]):
                n, d, dtn = get_neighbors(i, j)
                dist_matrix[i + reach, j + reach] = np.mean(d)

    avg_distance_matrix()
    return dist_matrix


def show_kld_plots(datasets, distributions, scores):
    img = np.zeros((len(datasets), len(distributions)))

    print(img.shape)

    for i in range(len(datasets)):
        for j in range(len(distributions)):
            img[i, j] = scores[i][j]

    """ 2D diagrams (deltas) """
    plt.figure(figsize=(23, 13))
    plt.subplot(311)
    plt.imshow(img.T, cmap='rainbow', aspect='auto', interpolation='None')
    plt.colorbar()

    plt.yticks(range(len(distributions)), list(map(lambda x: x.name, distributions)))
    plt.xticks(range(len(datasets)),
               datasets, fontsize=6, rotation='vertical')
    # plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig('./distributions.png', dpi=300)


def compute_show_best_distributions(datasets, distributions, subsampling = 1):

    import json

    results = {}

    with open('./dist_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    kld_array = []
    bestklds = []

    for dataset in tqdm(datasets):

        print(col(dataset, 'green'))

        dists, params, klds = dataset_range_hist_fit_distributions(
            dataset, distributions, 'all', subsampling
        )

        klds = list(map(lambda x: np.log(x), klds))
        minkld = np.min(klds)
        klds = np.array(klds)
        # klds[klds == minkld] = -1

        kld_array.append(klds)
        best_kld = np.array(dists)[klds == minkld][0]
        bestklds.append(best_kld)

        print(best_kld)

        results[dataset] = {
            'best_kld': best_kld.name,
            'val': minkld
        }
        # results[dataset]['klds'] = klds
        # results[dataset]['best_kld'] = best_kld

        with open('./dist_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    show_kld_plots(datasets, distributions, kld_array)


def compute_show_dispersions(datasets, ss=1):
    dispersions = []

    for dataset in tqdm(datasets):
        print(col(dataset, 'green'))
        dmed = find_moments_variation_for_dataset(dataset, ss)['dispkld']
        dispersions.append(dmed)

    dispersions = np.array(dispersions)
    datasets = np.array(datasets)
    dispersions_ = dispersions[np.logical_not(np.isnan(dispersions))]
    datasets = datasets[np.logical_not(np.isnan(dispersions))]
    dispersions = dispersions_

    dispersions, datasets = zip(*sorted(zip(dispersions, datasets)))

    img = np.zeros((len(dispersions), 1))
    img[:, 0] = dispersions
    plt.imshow(img.T, cmap='rainbow', aspect='auto', interpolation='None')
    plt.xticks(range(len(datasets)),
               datasets, fontsize=6, rotation='vertical')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# def get_increments(W):
#     from scipy.signal import convolve
#     return convolve(W, [-1, 1], mode='valid')
#
# def get_cdata_increments(cdata):
#     return np.apply_along_axis(lambda x: get_increments(x), axis=-1, arr=np.real(cdata)), np.apply_along_axis(lambda x: get_increments(x), axis=-1, arr=np.imag(cdata))
#
# def get_increment_distribution_for_dataset(dataset):
#
#     cdata = dataset_to_cdata(dataset)
#
#     real_cdata, imag_cdata = np.real(cdata), np.imag(cdata)
#
#     real_incs = np.apply_along_axis(lambda x: get_increments(x), axis=-1, arr=real_cdata)
#     imag_incs = np.apply_along_axis(lambda x: get_increments(x), axis=-1, arr=imag_cdata)
#
#     plt.hist(real_incs.flatten(), bins='auto', fc=(1, 0, 0, 0.3))
#     plt.hist(imag_incs.flatten(), bins='auto', fc=(0, 1, 0, 0.3))
#     plt.hist(real_cdata.flatten(), bins='auto', fc=(0, 0, 1, 0.3))
#     plt.hist(imag_cdata.flatten(), bins='auto', fc=(1, 1, 0, 0.3))
#     plt.show()
#
#     show_fitted_distributions_for_cdata_and_range()

def fit_dist(dist, data, rng):
    res = None
    if isinstance(rng, int):
        res = dist.fit(data, loc=0)
    elif rng == 'all':
        # res = dist.fit(np.reshape(data, (data.shape[0] * data.shape[1], -1)), loc=0)
        res = dist.fit(data.flatten())
    x = np.linspace(np.min(data), np.max(data), 100)

    arg = res[:-2]
    loc = res[-2]
    scale = res[-1]

    return loc, scale, arg
    # x = np.linspace(np.min(data), np.max(data), 1000)
    # y = dist.pdf(x, *arg, loc, scale) if arg else dist.pdf(x, loc, scale)
    # plt.hist(data.flatten(), bins='auto', density=1)
    # plt.plot(x, y, label=dist.name)
    # plt.legend(loc='upper right')
    # plt.show()


from scipy.special import kv
""" fonction de bessel de deuxième espèce modifiée """

class k_gen(stats.rv_continuous):

    def _pdf(self, x, mu, alpha, beta):
        return 2/((gamma(alpha) * gamma(beta))) * ((alpha * beta)/mu) ** ((alpha+beta)/2) * x ** ((alpha+beta)/2 - 1) * kv(alpha-beta, 2*np.sqrt((alpha*beta*x)/mu))

    def _argcheck(self, mu, alpha, beta):
        return mu > 0 and alpha > 0 and beta > 0

k = k_gen(name='k', a=0, b=np.inf)

def exponential_family():
    return [
        stats.lognorm, stats.gamma, stats.chi2, stats.beta, stats.weibull_min, stats.invgamma, stats.geninvgauss, k
    ]

# ds: list[Any] = list(filter(lambda x: x != '10_013_CStFA', load_datasets('C')))
# show_fitted_distributions_for_dataset_and_range('02_002_CStFA', exponential_family(), 'all')

from radarpkg2.processing.dataset_to_cdata import get_datasets

from scipy.stats import norm, cauchy, t, exponpow, gennorm, genhyperbolic, hypsecant, logistic, fisk


# fit_dist(t,
#          get_cdata_increments(
#              dataset_to_cdata('06_088_CStFA'))[0]
#          , 'all')

# fit_dist(t, np.imag(dataset_to_cdata('10_104_TTrFA')), 'all')

"finite mixture models"

""" 06_084_CStFA 26 degrés de liberté -> presque gaussien """
""" 06_086_CStFA gaussien """
""" 11_009_CStFA 9 degrés """
""" 11_015_CStFA 14 degrés """

# cdata = dataset_to_cdata('02_027_CStFA')
#
# means, meds, stds = [], [], []
# for i in range(cdata.shape[1]):
#     res = cdata_to_range_hist(cdata, i)
#     means.append(res[0])
#     meds.append(res[1])
#     stds.append(res[2])

# plt.plot(meds, label='median')
# plt.plot(means, label='mean')
# plt.plot(stds, label='std')
# plt.legend()
# plt.show()

# compute_show_dispersions(datasets)
# plt.savefig('./klds.png', dpi=300)

# compute_show_best_distributions(ds)
# show_fitted_distributions_for_dataset_and_range('02_005_CStFA', get_dist_list(), 'all')

# print(res.shape)
#
# [meds, means, stds] = np.array(res).T
# get_increment_distribution_for_dataset('02_001_CStFA')

# plt.plot(skew, label='skew')
# plt.plot(kurt, label='kurtosis')
# plt.legend(loc='upper left')
# plt.show()

# mat = make_kld_matrix_for_ranges('02_003_CStFA')
# mat = make_kld_matrix_for_ranges('08_059_CStFA')
# plt.imshow(mat, aspect='auto', cmap='bone', interpolation='None')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# print(compute_kld_for_kde_and_fit_distribution('00_005_TTrFA', stats.weibull_min, 20))
# print(x,y)
# plt.plot(x,y)
# plt.show()

# M = dataset_to_neighbor_distance_map('00_005_TTrFA', 4, accuracy=2)
# M = dataset_to_neighbor_distance_map('08_059_CStFA', 8, accuracy=1)
# M = dataset_to_neighbor_distance_map('06_092_CStFA', 4, accuracy=3)

# plt.imshow(M, aspect='auto', cmap='Reds', interpolation='None')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

""" faire boites à moustaches pour toutes les données """
""" montrer la déviation minimale des kld """

""" variations des distributions du clutter suivant range """
""" lois qui sont dans la famille exponentielle """
""" trouver représentant dans la famille expo """
""" cfar exponentiel sur les données suivent une distribution """
""" tester distribution k """
""" radar image prior """
""" tester dist multivariées """

""" tableau kld = kld range i vs tous les ranges puis tableau de variation de la kld (q3-q1)/(q3+q1) """
