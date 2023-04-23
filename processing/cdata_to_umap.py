import numpy as np
import umap
import umap.plot
from radarpkg.visualisation.visualisation import colorize


def red(text):
    return "\033[1;91m " + text + "\033[0m"

def cdata_reshape(data):
    n_samples = data.shape[0] * data.shape[1]
    colors = colorize(data[:, :, :].reshape((n_samples, -1)))
    real_part = np.real(data[..., 0:])
    im_part = np.imag(data[..., 0:])
    final_data = np.concatenate([real_part, im_part],
                                axis=-1)
    sample_dim = final_data.shape[0] * final_data.shape[1]
    formatted_data = final_data.reshape((sample_dim, final_data.shape[-1]))
    return np.real(formatted_data), colors


def cdata_to_umap(data, n_neighbors, min_dist, metric='euclidean', low_memory=False, verbose=0, distances=None):

    try:
        formatted_data, colors = cdata_reshape(data)
        print(colors.shape)
    except IndexError as e:
        print('Invalid data, skipping dataset' + repr(e))
    except TypeError as e:
        print('Invalid data, skipping dataset' + repr(e))
    except AttributeError as e:
        print(repr(e))

    else:

        # try:
            if verbose:
                print('mapping...')

            if metric != 'precomputed':
                mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, low_memory=low_memory,
                                   verbose=verbose).fit(
                    formatted_data)
            else:
                mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='precomputed', low_memory=low_memory,
                                   verbose=verbose).fit(
                    distances)
                return mapper, colors

            if verbose:
                print('mapping done')
            return mapper, colors

        # except ValueError as e:
        #     print('Invalid data, skipping dataset' + repr(e))
        #     return None
