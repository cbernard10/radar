Dépendances :
    
    pip install numpy scipy umap-learn[plot] hdbscan numba scikit-learn opencv-python

Les données CSIR et le script principal main.py doivent être situés au même niveau que radarpkg:

    |- csir
    |   |- 00_005_TTrFA
    |   |- 00_010_TTrFA
    |   |- ...
    |
    |- radarpkg
    |- main.py

Exemples d'utilisation :

**Coefficients de réflexion**

    from radarpkg2.processing.dataset_to_cdata import dataset_to_cdata
    from radarpkg2.processing.cdata_to_burg import cdata_to_burg
    from radarpkg2.visualisation.visualisation import plot_burg_all

    if __name__ == '__main__':

        dataset = '00_005_TTrFA'
        cdata = dataset_to_cdata(dataset)
        burg = cdata_to_burg(cdata, dataset, order = 6, gamma = 0.01)
    
        plot_burg_all(burg, rows=2, cols=3)

**CFAR**

    import matplotlib.pyplot as plt
    import numpy as np
    
    from radarpkg2.processing.dataset_to_cdata import dataset_to_cdata
    from radarpkg2.processing.cdata_to_fft import cdata_to_fft
    from radarpkg2.processing.CFAR import CACFAR, GOCFAR, SOCFAR, OSCFAR
    from radarpkg2.visualisation.visualisation import show_subplots
    
    if __name__ == '__main__':
    
        dataset = '00_005_TTrFA'
        cdata = dataset_to_cdata(dataset, subsampling=1, normalized=False)
        doppler = np.real(cdata_to_fft(cdata, 200))
    
        PFA = 1e-6
        n_control = 17
        n_guard = 4
        clutter_level = 24 # only used for OSCFAR
    
        show_subplots(2, 2, [
            CACFAR(doppler, n_control, n_guard, PFA),
            GOCFAR(doppler, n_control, n_guard, PFA),
            SOCFAR(doppler, n_control, n_guard, PFA),
            OSCFAR(doppler, n_control, n_guard, clutter_level, PFA),
        ], plt_type='imshow', cmap='bone', titles=['CACFAR', 'GOCFAR', 'SOCFAR', 'OSCFAR'], aspect='auto')
        plt.show()

**Comparaison des coefficients de régularisation**

    from radarpkg2.processing.dataset_to_cdata import dataset_to_cdata
    from radarpkg2.processing.cdata_to_burg import cdata_to_burg_spectrum
    import numpy as np
    from radarpkg2.visualisation.visualisation import animate_array
    from tqdm import tqdm
    
    if __name__=='__main__':
    
        dataset = '04_022_TTrFA'
        cdata = dataset_to_cdata(dataset, subsampling=1, normalized=1)
    
        arr = [
            np.hstack(
                [cdata_to_burg_spectrum(cdata, i, 6, 64, j) for j in np.logspace(-9, 2, 10)]
            ) for i in tqdm(range(cdata.shape[0]))]
        #
        animate_array(arr, w = 2200, h = 1200)
