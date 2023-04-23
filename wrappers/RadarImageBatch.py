from RadarImage import RadarImage
import matplotlib.pyplot as plt
from printer import Printer
from ..processing.dataset_to_cdata import get_datasets
from tqdm import tqdm
from multiprocessing import Pool
from ..processing.student_protocol import dataset_to_student_params, kld_params, karcher_mean, geodesic_between_start_end_pts, plot_geodesic, cdata_ranges_to_student_params
import numpy as np
from ..visualisation.visualisation import red, yellow

class RadarImageBatch:

    def __init__(self, datasets=None, resolution=1):

        self.datasets = datasets if datasets else get_datasets()
        self.radarImages = []
        print(yellow('loading datasets...'))
        for ds in tqdm(self.datasets):
            self.radarImages.append(RadarImage(ds, resolution))

    def box_plots(self):

        fig, ax = plt.subplots()
        data = []

        for img in self.radarImages:
            data.append(np.real(img.cdata.flatten()))

        ax.boxplot(data)
        plt.show()

    def compute_linked_geodesics_between_datasets(self, xatol, fatol, dt, plot=False):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.set_xlim3d(-0.5, 0.5)
        ax.set_xlabel('μ')
        ax.set_ylabel('σ')
        ax.set_zlabel('ν')

        for i in range(len(self.datasets) - 1):

            print(f"computing geodesic between {self.datasets[i]} and {self.datasets[i+1]}")
            start = self.radarImages[i]
            end = self.radarImages[i+1]

            X, Y, Z, dX, _ = start.compute_geodesic(end, xatol, fatol, dt)

            ax.plot(X, Y, Z)

        plt.show()

    def compute_geodesic_net(self, xatol, fatol, dt):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.set_xlim3d(-0.5, 0.5)
        ax.set_xlabel('μ')
        ax.set_ylabel('σ')
        ax.set_zlabel('ν')

        n = len(self.datasets)
        net_size = n*(n-1)/2
        k=0

        for i in range(len(self.datasets) - 1):
            for j in range(len(self.datasets) - 1):

                if i>=j: continue
                else:

                    k = k+1
                    print(f"computing geodesic between {self.datasets[i]} and {self.datasets[j]}")
                    start = self.radarImages[i]
                    end = self.radarImages[j]

                    X, Y, Z, dX, _ = start.compute_geodesic(end, xatol, fatol, dt)

                    print(f"{k/net_size * 100}%")
                    ax.plot(X, Y, Z)

        plt.show()

    def compute_geodesic_net_parallel(self, xatol, fatol, dt):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.set_xlim3d(-0.5, 0.5)
        ax.set_zlim3d(0, 32)
        ax.set_xlabel('μ')
        ax.set_ylabel('σ')
        ax.set_zlabel('ν')

        with Pool() as p:
            result = p.map(dataset_to_student_params, self.datasets)

        result = [r['Real'] for r in result]
        print(result)
        args = [(result[i], result[i+1], xatol, fatol, dt) for i in range(len(self.datasets) - 1)]
        result=[]

        with Pool() as p:
            result = p.starmap(geodesic_between_start_end_pts, args)

        for res in result:
            X, Y, Z, _, _ = res
            ax.plot(X, Y, Z)

        plt.show()

if __name__ == '__main__':

    # ds = get_datasets()[:100]
    ds = ['00_005_TTrFA', '00_010_TTrFA', '00_011_TTrFA', '00_016_TTrFA', '00_025_TTrFA']
    # ds = ['05_001_CStFA', '05_002_CStFA', '05_003_CStFA', '05_004_CStFA', '05_005_CStFA', '05_008_CStFA', '05_009_CStFA', '05_010_CStFA', '05_011_CStFA', '05_012_CStFA', ]

    batch = RadarImageBatch()
    # batch.box_plots()
    batch.compute_geodesic_net(1e-4, 1e-10, 1e-2)

