from dataset_to_cdata import dataset_to_cdata, cdata_to_amplitude
import numpy as np
from PIL import Image

def sampler(dataset, n_samples, size):

    samples = []

    X = dataset_to_cdata(dataset, 1, 1)

    x_max, y_max = X.shape[0]-size, X.shape[1]-size

    origins_x = np.random.randint(0, x_max, n_samples)
    origins_y = np.random.randint(0, y_max, n_samples)

    for ox, oy in zip(origins_x, origins_y):
        samples.append(
            X[ox:ox+size, oy:oy+size]
        )

    return samples

def sample_and_save(dataset, n_samples, size, path='.'):

    samples = sampler(dataset, n_samples, size)
    for i, s in enumerate(samples):
        amp = cdata_to_amplitude(s)
        amp = amp - np.min(amp)
        amp = amp / np.max(amp) * 255

        im = Image.fromarray(amp)
        im = im.convert('L')
        im.save(path + f'/sample{i}.png')
