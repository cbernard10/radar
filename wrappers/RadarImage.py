# from dataset_to_cdata import dataset_to_cdata
from glob import glob
import scipy.io as sio
import numpy as np
from ..processing.dataset_to_burg import cdata_to_burg_jit
from ..processing.burg_to_umap import burg_to_umap, dataset_to_burg_to_umap_save
from ..processing.cdata_to_umap import cdata_to_umap
from ..processing.student_protocol import cdata_to_student_params, kld_params, karcher_mean, geodesic_between_start_end_pts, plot_geodesic, cdata_ranges_to_student_params, plot_pts_and_mean
from ..processing.umap_to_hdbscan import umap_to_hdbscan
from ..processing.clear_folders import clear_burg_folder
import matplotlib.pyplot as plt
from ..processing.dataset_to_summary import load_summary
from ..processing.dataset_to_cdata import dataset_to_cdata

class RadarImage:

    def __init__(self, dataset, resolution=1):

        self.dataset = dataset.strip()
        self.resolution = resolution
        self.cdata = None
        self.n_azimuth, self.n_range, self.n_pulses = [None, None, None]
        self.fft = None
        self.burg = None
        self.burg_umap = None
        self.cdata_umap = None
        self.student_params = None
        self.type = self.dataset[7]
        self.ranges_params = None
        self.kld_matrix = None
        self.initial_speed = None
        self.burg_hdbscan = None
        self.cdata_hdbscan = None
        self.pow = None
        self.gamma = 0
        self.v_geodesic = None

    def box_plot(self):
        plt.boxplot(np.real(self.cdata).flatten())

    def box_plot_ranges(self):
        fig, ax = plt.subplots()

        data = []

        for i in range(self.n_range):
            data.append(np.real(self.cdata[i,...]).flatten())

        ax.boxplot(data)
        plt.show()

    def clear_folder(self):
        clear_burg_folder(self.dataset)

    def compute_cacfar(self):
        return 0

    def compute_cdata(self, resolution=1, clear_cache=False):
        if self.cdata is not None and not clear_cache: return self.cdata
        self.cdata = dataset_to_cdata(self.dataset, resolution)
        self.n_azimuth, self.n_range, self.n_pulses = self.cdata.shape
        return self.cdata

    def compute_pow(self, show=1):

        res = []
        if self.pow is not None:
            res = self.pow
        else:
            I = np.amax(np.real(self.compute_cdata()), -1)
            Q = np.amax(np.imag(self.compute_cdata()), -1)
            res = 10*np.log(10* (I**2 + Q**2))
        if show:
            plt.imshow(res.T[::-1], aspect='auto', cmap='bone', interpolation='nearest')
            plt.show()
        return res

    def compute_fft(self, clear_cache = False):

        if self.fft is not None and not clear_cache: return self.fft
        else:
            data = np.array([self.compute_cdata()[:, i].ravel() for i in range(self.compute_cdata().shape[1])])[:, :]
            range_doppler = np.fft.fftshift(np.fft.fft(data), axes=(1,))
            self.fft = range_doppler
            return self.fft

    def compute_burg(self, order = 6, clear_cache = False):

        if self.burg is not None and not clear_cache: return self.burg
        else:
            res = cdata_to_burg_jit(self.compute_cdata(), self.dataset, order, self.gamma)
            self.burg = res
            return self.burg

    def compute_complex_geodesic(self, xatol, fatol, dt, plot=True, clear_cache=False):
        if self.v_geodesic is not None and not clear_cache:
            X, Y, Z, dX, _ = self.v_geodesic
            if plot:
                plot_geodesic(X, Y, Z)
            return self.v_geodesic
        else:
            params = self.compute_student_params()
            res = geodesic_between_start_end_pts(params['Real'],
                                                 params['Imag'], xatol, fatol, dt)
            X, Y, Z, dX, _ = res
            self.v_geodesic = res
            if plot:
                plot_geodesic(X, Y, Z)
            return self.v_geodesic

    def ds_to_bu_to_um_save(self, order=6, n_neighbors=100, min_dist=0, metric='euclidean',
                                 save=True, s=0.01, subPath='' ):

        dataset_to_burg_to_umap_save(self.dataset, self.resolution, order, n_neighbors, min_dist, metric, save, s, subPath)

    def compute_burg_umap(self, clear_cache = False, n_neighbors=100, min_dist=0, metric='euclidean', low_memory=False, verbose=0):

        if self.burg_umap is not None and not clear_cache: return self.burg_umap
        else:
            res = burg_to_umap(self.compute_burg(), n_neighbors, min_dist, metric, low_memory, verbose)
            self.burg_umap = res
            return self.burg_umap

    def compute_cdata_umap(self, clear_cache = False, n_neighbors=100, min_dist=0, metric='euclidean', low_memory=False, verbose=0):

        if self.cdata_umap is not None and not clear_cache: return self.cdata_umap
        else:
            res = cdata_to_umap(self.compute_cdata(), n_neighbors, min_dist, metric, low_memory, verbose)
            self.cdata_umap = res
            return self.cdata_umap

    def compute_burg_hdbscan(self, clear_cache = False, save=0, verbose=0):
        if self.burg_hdbscan is not None and not clear_cache: return self.burg_hdbscan
        else:
            res = umap_to_hdbscan(self.compute_burg_umap(verbose=1), save=save, dataset=self.dataset, filename='umap_1', verbose=verbose)
            self.burg_hdbscan = res
            return self.burg_hdbscan

    def compute_cdata_hdbscan(self, clear_cache = False, save=0, verbose=0):
        if self.cdata_hdbscan is not None and not clear_cache: return self.cdata_hdbscan
        else:
            res = umap_to_hdbscan(self.compute_cdata_umap(verbose=1), save=save, dataset=self.dataset, filename='umap_1', verbose=verbose)
            self.cdata_hdbscan = res
            return self.cdata_hdbscan

    def compute_student_params(self, clear_cache = False):

        if self.student_params is not None and not clear_cache: return self.student_params
        else:
            res = cdata_to_student_params(self.compute_cdata())
            self.student_params = res

        return self.student_params

    def difference(self, radarImage, mode='Real'):
        if not isinstance(radarImage, RadarImage):
            print('radarImage must be a RadarImage')
            return None
        else:
            return kld_params(self.compute_student_params()[mode], radarImage.compute_student_params()[mode])

    def karcher_mean(self, radarImage, pow=2, weights=[1, 1], mode='Real'):
        if not isinstance(radarImage, RadarImage):
            print('radarImage must be a RadarImage')
            return None
        else:
            return karcher_mean([self.compute_student_params()[mode], radarImage.compute_student_params()[mode]], pow=pow, weights=weights)

    def compute_geodesic(self, radarImage, xatol, fatol, dt, mode='Real', plot=False):
        if not isinstance(radarImage, RadarImage):
            print('radarImage must be a RadarImage')
            return None
        else:
            res = geodesic_between_start_end_pts(self.compute_student_params()[mode], radarImage.compute_student_params()[mode], xatol, fatol, dt)
            X, Y, Z, dX, _ = res
            self.initial_speed = dX
            if plot:
                plot_geodesic(X, Y, Z)
            return res

    def compute_range_params(self, clear_cache=False, labels=False, paths=False, show=1):
        if self.ranges_params is not None and not clear_cache: return self.ranges_params
        else:
            self.ranges_params = cdata_ranges_to_student_params(self.compute_cdata())
            if show:
                plot_pts_and_mean(self.ranges_params, labels=labels, paths=paths)
            return self.ranges_params

    def compute_kld_matrix_for_ranges(self, clear_cache=False):
        nr = self.n_range
        if self.kld_matrix is not None and not clear_cache: return self.kld_matrix
        else:
            mat = np.zeros((nr, nr))
            for i in range(nr):
                for j in range(nr):
                    mat[i,j] = kld_params(self.compute_range_params()[i], self.compute_range_params()[j])

        self.kld_matrix = mat
        return self.kld_matrix

    def process_all(self, reset = 0):
        if reset: clear_burg_folder(self.dataset)
        self.compute_cdata()
        self.compute_burg()
        # self.ds_to_bu_to_um_save()

    def plot_ranges(self):
        plot_pts_and_mean(self.compute_range_params())

if __name__ == '__main__':

    ri1 = RadarImage('08_075_CScFA', 1)
    ri1.compute_complex_geodesic(1e-4, 1e-12, 1e-2)
    # ri1.compute_range_params(labels=1, paths=1)

    # ri1.box_plot_ranges()
    # ri1.compute_hdbscan(save=1, verbose=1)
    # ri1.plot_ranges()