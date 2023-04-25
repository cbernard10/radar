#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pylab as plt
from colorsys import hls_to_rgb, hsv_to_rgb

from numpy import pi
import cv2
import os
from os import unlink

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_subplots(rows, cols, data, axis='on', figsize=(6, 6), plt_type='plot', X=None, Y=None, titles=None, vmin=None,
                  vmax=None, cmap=None, cmaps=None, s=1, contour_data=None, interpolation='nearest', aspect='auto'):


    fig = plt.figure(figsize=figsize)
    for i, d in enumerate(data):
        if plt_type == 'plot':
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.plot(d)
        if plt_type == 'scatter':
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.scatter(d[:, 0], d[:, 1], s=s)
            ax.tick_params(labelbottom=False, labelleft=False)
        if plt_type == 'imshow':
            ax = fig.add_subplot(rows, cols, i + 1)
            if cmaps:
                ax.imshow(d, cmap=cmaps[i], vmin=vmin, vmax=vmax, interpolation=interpolation, aspect=aspect)
            else:
                ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, aspect=aspect)
            if titles:
                ax.set_title(titles[i])
            ax.axis(axis)
            if contour_data:
                ax.contour(np.abs(contour_data[i][::-1]), 20, colors=['white'], linewidths=1)
                ax.contour(np.angle(contour_data[i][::-1]), 20,
                           colors=['white'], linewidths=1, antialiased=True)
        if plt_type == 'surface':
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            ax.plot_surface(X, Y, d)
        if plt_type == 'wireframe':
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            ax.plot_wireframe(X, Y, d)
    return fig

def colorize(z):

    """ transforme un array-like complexe z en couleurs hls """

    r = np.abs(z)
    arg = np.angle(z)

    a = 0.5

    h = (arg / (2 * pi) + 0.5)
    l = (1 - a ** r)
    s = 1

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)
    c = c.swapaxes(0, 2)
    return c


def colorize_1d(z):
    r = np.abs(z)
    arg = np.angle(z)

    a = 0.5

    h = (arg / (2 * pi) + 0.5)
    l = (1 - a ** r)
    s = 1

    c = np.vectorize(hls_to_rgb)(h, l, s)

    return np.array(c)


def show_multi(rows, cols, data, figsize=(6, 6), titles=None, aspect='auto', interpolation='nearest', axis='on'):
    """ affiche les éléments complexes de data dans l'ordre sur rows lignes et cols colonnes """

    fig = plt.figure(figsize=figsize)
    for i, d in enumerate(np.transpose(data, (2, 0, 1))):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = colorize(d)
        ax.imshow(img, aspect=aspect, interpolation=interpolation)
        if titles:
            ax.set_title(titles[i])
        ax.axis(axis)


def complex_scatter(dataset, path, subsampling=1, save=True):
    """ affiche les nombres complexes en nuage de points, colorés selon leur affixe """

    dataset = dataset.ravel()[::subsampling]

    colors = colorize_1d(dataset).T

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.scatter(np.real(dataset), np.imag(dataset), c=colors, s=0.5)
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax.add_artist(circle)
    plt.axes().set_aspect('equal')

    if save:
        plt.savefig(path)

def plot_burg(burg_output, ranges=None, times=None, subsampling=1, s=0.05):

    """ affiche les sor"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if ranges and times:

        ax.imshow(colorize(burg_output), extent=[np.min(times), np.max(times), np.min(ranges), np.max(ranges)],
                     aspect='auto', interpolation='none')
        ax.set(xlabel="time (s)", ylabel="range (m)")

    else:
        ax.imshow(colorize(burg_output),
                 aspect='auto', interpolation='none')
    # plt.show()
    return fig, ax

def plot_burg_all(burg_output, shape=(2, 3), figsize=(6, 6)):

    """ affiche les coefficients de réflexion de chaque ordre """

    rows, cols = shape
    n_orders = burg_output.shape[-1]
    fig = plt.figure(figsize=figsize)
    for i in range(n_orders):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(colorize(burg_output[...,i]),
         aspect='auto', interpolation='none')

    plt.show()

def map_and_scatter(burg_output, ranges, times, subsampling=1, s=0.05):
    """ affiche l'image complexe à gauche et le nuage de points à droite """

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(colorize(burg_output), extent=[np.min(times), np.max(times), np.min(ranges), np.max(ranges)],
                 aspect='auto', interpolation='none')

    ax[0].set(xlabel="time (s)", ylabel="range (m)")

    dataset = burg_output.ravel()[::subsampling]
    colors = colorize_1d(dataset).T
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax[1].set_xlim([-1.05, 1.05])
    ax[1].set_ylim([-1.05, 1.05])
    ax[1].scatter(np.real(dataset), np.imag(dataset), c=colors, s=s, marker='.')
    ax[1].add_artist(circle)
    ax[1].set_aspect('equal')

    plt.show()
    return fig, ax

def make_scatter(coeffs, size=(8, 8), dpi=300, s=0.1, path=None):

    fig, ax = plt.subplots(1, 1, figsize=size)
    coeffs = coeffs.ravel()[::1]
    colors = colorize_1d(coeffs).T
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.scatter(np.real(coeffs), np.imag(coeffs), c=colors, s=s)
    ax.add_artist(circle)
    ax.set_aspect('equal')
    if path:
        plt.savefig(path, dpi=dpi)
    plt.show()
    plt.close(fig)


def umap_plot(umap_data, order = 0, s = 1):

    mapper = umap_data[0]
    colors = umap_data[1]

    c = colors[order].squeeze()

    plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=c,
                                s=s, alpha=1)

    plt.show()
    plt.savefig('./umap_output.png', dpi=300)

def red(text):
    return "\033[1;91m " + str(text) + "\033[0m"

def green(text):
    return "\033[1;92m " + str(text) + "\033[0m"

def yellow(text):
    return "\033[1;93m " + str(text) + "\033[0m"

def blue(text):
    return "\033[1;94m " + str(text) + "\033[0m"

def col(text, color):
    c = {'red': "\033[1;91m ", 'green': "\033[1;92m ", 'yellow': "\033[1;93m ", 'blue': "\033[1;94m "}[color]
    return c + str(text) + "\033[0m"

def animate_array(array, w, h):

    ts = 50
    i=0
    play = True
    while True:

        if not play:
            i -= 1

        img = array[i]
        img_w, img_h = img.shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('cfar', img)

        i= (i+1) % len(array)
        k = cv2.waitKey(ts)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        if k == ord('p'):
            play = not(play)
        if k == ord('e'):
            ts = max(round(ts / 2), 1)
        if k == ord('a'):
            ts = min(ts * 2, 1024)

def convert_to_jpeg(im_path):

    with Image.open(im_path + '.png') as im:
        im.convert('RGB').save(im_path + '.jpg', 'JPEG')

    unlink(im_path + '.png')


def umap_to_images(umap_data, dataset, filename, s=1, subFolder=''):
    if not umap_data:
        print('no umap data')
        return None

    root = 'csir/'
    path = root + dataset
    mapper = umap_data[0]
    colors = umap_data[1]
    power_colors = umap_data[2]

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
        # with open(path + '/' + dataset + '_umap.json', 'w', encoding='utf-8') as outputFile:
        #     json.dump({'data': mapper.embedding_.tolist(), 'color': colors.tolist()}, outputFile, indent=4,
        #               allow_nan=True)

        for i in range(colors.shape[0] + 1):
            fig = plt.figure()
            if i == 0:
                plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=power_colors.squeeze(), cmap='jet',
                            s=s, marker=',', alpha=1)
            else:
                plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=colors[i - 1].squeeze(),
                            s=s, marker=',', alpha=1)

            plt.savefig(
                path + '/burg/umap/' + subFolder + '/' + filename + '_' + str(i) + '.png',
                bbox_inches='tight', dpi=200)
            plt.close(fig)

            # im_path = '../csir/00_005_TTrFA/burg/umap/barb/u_1'
            # im = Image.open(im_path + '.png')
            # im.convert('RGB').save(im_path + '.jpg', 'JPEG')

            im_path = path + '/burg/umap/' + subFolder + '/' + filename + '_' + str(i)
            convert_to_jpeg(im_path)

            # fig = plt.figure()
            # plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], c=colors[1].squeeze(),
            #             s=s)
            #
            # plt.savefig(
            #     path + '/burg/umap/' + subPath + filename + '_' + str(1),
            #     bbox_inches='tight', dpi=300)
            # plt.close(fig)



def save_burg_maps(burg_output, dataset):

    order = burg_output.shape[-1]
    root = 'csir/'
    path = root + dataset

    try:
        os.mkdir(path + '/burg/')
    except FileExistsError:
        pass
    try:
        os.mkdir(path + '/burg/umap/')
    except FileExistsError:
        pass

    for i in range(order):

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if i == 0:
            ax.imshow(np.real(burg_output[:, ::-1, 0].T), aspect='auto', interpolation='none', cmap='jet')
            ax.set(xlabel="time (s)", ylabel="range (m)")
        else:
            ax.imshow(colorize(burg_output[:, ::-1, i]),
                      aspect='auto', interpolation='none')
            ax.set(xlabel="time (s)", ylabel="range (m)")

        plt.savefig(path + '/burg/' + 'c' + str(i) + '.png', bbox_inches='tight', dpi=200)

        # plt.close(fig)
        # return fig, ax