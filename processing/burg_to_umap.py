import json
import os
from os import unlink

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
from PIL import Image

from radarpkg2.processing.dataset_to_burg import dataset_to_burg_jit
from radarpkg2.processing.dataset_to_cdata import dataset_to_cdata
from radarpkg2.processing.cdata_to_burg import cdata_to_burg
from ..visualisation.visualisation import colorize, convert_to_jpeg, red


def burg_reshape(burg_data):

    """ reshapes the burg coefficients for the input of umap """

    n_samples = burg_data.shape[0] * burg_data.shape[1]
    colors = colorize(burg_data[:, :, 1:].reshape((n_samples, -1)))
    power_colors = burg_data[:, :, 0, np.newaxis].reshape((n_samples, -1))
    real_part = np.real(burg_data[..., 1:])
    im_part = np.imag(burg_data[..., 1:])
    final_data = np.concatenate([burg_data[..., 0, np.newaxis] / np.max(burg_data[..., 0]), real_part, im_part],
                                axis=-1)
    sample_dim = final_data.shape[0] * final_data.shape[1]
    formatted_burg_data = final_data.reshape((sample_dim, final_data.shape[-1]))
    return np.real(formatted_burg_data), colors, power_colors

def burg_to_umap(burg_data, n_neighbors, min_dist, metric='euclidean', low_memory=True, verbose=0, save=0):

    """ applique UMAP sur les coefficients de réflexion """

    try:
        formatted_data, colors, power_colors = burg_reshape(burg_data)
    except IndexError as e:
        print('Invalid data, skipping dataset' + repr(e))
    except TypeError as e:
        print('Invalid data, skipping dataset' + repr(e))
    except AttributeError as e:
        print(repr(e))
    else:
        try:
            if verbose:
                print('mapping...')
            mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, low_memory=low_memory,
                               verbose=verbose).fit(
                formatted_data)

            if verbose:
                print('mapping done')
            return mapper.embedding_, colors, power_colors
        except ValueError as e:
            print('Invalid data, skipping dataset' + repr(e))
            return None


# def dataset_to_burg_to_umap(dataset, subsampling=1, order=4, n_neighbors=10, min_dist=0, metric='euclidean'):
#     print('\n' + "\033[1;92m dataset: " + dataset + "\033[0m")
#
#     try:
#         burg_data = dataset_to_burg_jit(dataset, subsampling, order)
#         mapper, colors, power_colors = burg_to_umap(burg_data, n_neighbors, min_dist, metric)
#     except BaseException as e:
#         print(repr(e))
#     else:
#         return [mapper, colors, power_colors]

def umap_to_images(umap_data, dataset: str, filename: str, s=0.01, subFolder=''):

    """ saves umap output as images """

    if not umap_data:
        print('no umap data')
        return None

    root = 'csir/'
    path = root + dataset
    embedding = umap_data[0]
    colors = np.real(umap_data[1])
    power_colors = np.real(umap_data[2])

    try:
        os.mkdir(path + '/burg/umap/')
    except FileExistsError:
        pass
    try:
        os.mkdir(path + '/burg/umap/' + subFolder)
    except FileExistsError:
        pass

    finally:
        print('saving')

        for i in range(colors.shape[0] + 1):
            fig = plt.figure()
            if i == 0:
                plt.scatter(np.real(embedding[:, 0]), np.real(embedding[:, 1]), c=power_colors.squeeze(), cmap='jet',
                            s=s, marker=',', alpha=0.1)
            else:
                plt.scatter(np.real(embedding[:, 0]), np.real(embedding[:, 1]), c=colors[i - 1].squeeze(),
                            s=s, marker=',', alpha=0.1)

            # print(path, subFolder, filename)
            plt.savefig(
                path + '/burg/umap/' + subFolder + '/' + filename + '_' + str(i) + '.png',
                bbox_inches='tight', dpi=200)
            plt.close(fig)

            im_path = path + '/burg/umap/' + subFolder + '/' + filename + '_' + str(i)
            convert_to_jpeg(im_path)


def dataset_to_burg_to_umap_save(dataset: str, subsampling=1, order=4, n_neighbors=10, min_dist=0, metric='euclidean',
                                 save=True, s=0.01, subPath=''):

    """ écrit la sortie de UMAP en json et png à partir du nom du jeu de données """

    root = 'csir/'
    path = root + dataset

    print('\n' + "\033[1;92m dataset: " + dataset + "\033[0m")

    try:
        cdata = dataset_to_cdata(dataset)
        burg_data = cdata_to_burg(cdata, dataset, order, gamma=0.001, progress=True)
        umap_data, colors, power_colors = burg_to_umap(burg_data, n_neighbors, min_dist, metric, verbose=1)
    except IndexError as e:
        print('Invalid data, skipping dataset' + repr(e))
    except TypeError as e:
        print('Invalid data, skipping dataset' + repr(e))
    else:
        if save:
            try:
                print(path + '/burg/umap/' + subPath)
                os.mkdir(path + '/burg/umap/' + subPath)
            except FileExistsError:
                pass
            finally:
                print('saving')
                with open(path + '/' + dataset + '_umap.json', 'w', encoding='utf-8') as outputFile:
                    json.dump({'data': umap_data.embedding_.tolist(), 'color': colors.tolist()}, outputFile, indent=4,
                              allow_nan=True)

                for i in range(min(order, 4)):
                    fig = plt.figure()
                    plt.scatter(umap_data.embedding_[:, 0], umap_data.embedding_[:, 1], c=colors[i].squeeze(),
                                s=s)
                    # plt.show(block=False)
                    plt.savefig(
                        path + '/burg/umap/' + subPath + 'n' + str(n_neighbors) +
                        'm' + str(metric) + 'c' + str(i) + 'o' + str(order) + '.png',
                        bbox_inches='tight', dpi=200)
                    plt.close(fig)

        return [umap_data.embedding_, colors]

# def coeffs_to_dict(coeffs):
#
#     return [{'x': i, 'y': j, 'coeffs': coeffs[i, j]} for i in range(coeffs.shape[0]) for j in range(coeffs.shape[1])]
