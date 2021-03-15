#!/usr/bin/env python
# coding: utf-8
import glob
import os, sys
import datetime
import argparse
import copy
import numpy as np
import h5py
import scipy.signal as signal

from pyhdf import SD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib
from config import *


def draw(data1, outjpg, titlename):
    data = data1.copy()
    # titlename = tmptitlename.format(YMD = time,band = "FOG")
    height, width = data.shape
    RR = 500.
    w, h = width / RR + 2., height / RR + 2.
    fig = plt.figure(figsize=[w, h])
    ax = plt.axes([1 / w, 1 / h, (w - 2) / w, (h - 2) / h])
    m = Basemap(llcrnrlon=105.0, llcrnrlat=13.0, urcrnrlon=135.0, urcrnrlat=43.0, \
                resolution='i', projection='cyl')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(13, 43.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
    m.drawmeridians(np.arange(105, 135.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])

    plt.xlabel('$Longitude$', fontsize=12, labelpad=24)
    plt.ylabel('$Latitude$', fontsize=12, labelpad=40)

    plt.title(titlename, fontsize=18)
    ffim = m.imshow(data[::-1], extent=[105, 135, 43, 13], vmax=40, vmin=0, cmap='rainbow')
    cax = plt.axes([1.2 / w, 1.3 / h, (w - 2) / 2.3 / w, (.25) / h])
    # *orientation * vertical or horizontal
    cbar = plt.gcf().colorbar(ffim, extend="both", orientation='horizontal', cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # L = TickLabels[band]
    tick = [x * 2 for x in range(0, 20)]
    cbar.set_ticks(tick)
    # cbar.set_ticklabels(L)

    plt.savefig(outjpg + 'FOG.png', dpi=200, bbox_inches = 'tight')
    plt.clf()
    plt.close(fig)


def plot(rat_fog, rat_cloud, outfc):
    # rat_fog = rat_fog1.copy()
    # rat_cloud = rat_cloud1.copy()

    xx = np.arange(rat_fog.shape[0])
    yy = np.arange(rat_fog.shape[1])
    # xx = np.arange(ProjectionMinLatitude, ProjectionMaxLatitude, ProjectionResolution)
    # yy = np.arange(ProjectionMinLongitude, ProjectionMaxLongitude, ProjectionResolution)
    x, y = np.meshgrid(yy, xx)
    rat_fog[rat_fog == 0] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    img1 = ax.imshow(rat_fog * 100, 'rainbow', vmin=0, vmax=30)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="3%", pad=0.1)

    contoursol = ax.contour(x, y, rat_cloud * 100, [10, 30, 50, 70, 100], colors='red', linewidths=0.3)
    ax.clabel(contoursol, fontsize=8, colors='red', fmt='%.0f')
    plt.colorbar(img1, cax=cax2)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(outfc + '.png', dpi=200, bbox_inches = 'tight')
    plt.close(fig)


def plot2(rat_fog, outfc):
    xx = np.arange(rat_fog.shape[0])
    yy = np.arange(rat_fog.shape[1])
    x, y = np.meshgrid(yy, xx)
    rat_fog[rat_fog == 0] = np.nan
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    img1 = ax.imshow(rat_fog * 100, 'rainbow', vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(img1, cax=cax2)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(outfc + '_cloud.png', dpi=200, bbox_inches = 'tight')

    plt.close(fig)

def ColorBar2Fog():
    colorlist = [
            [255, 255, 255],
            [234, 235, 255],
            [154, 153, 255],
            [50, 51, 255],
            [2, 36, 206],
            [5, 107, 111],
            [8, 179, 15],
            [110, 211, 9],
            [213, 242, 2],
            [254, 194, 0],
            [255, 92, 0],
            [255, 0, 0],
        ]

    clor = ["#%02x%02x%02x" % (i[0], i[1], i[2]) for i in colorlist]

    # cmap = matplotlib.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],[1., .8, 0.], [1., .4, 0.]])
    cmap = matplotlib.colors.ListedColormap(clor, 'indexed')
    # cmap.set_over((1., 0., 0.))
    # cmap.set_under((0., 0., 1.))

    # bounds = [-1., -.5, 0., .5, 1.]
    bounds = [0.0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


def ColorBar2Cld():
    colorlist = [
            [252, 252, 252],
            # [233, 233, 233],
            [219, 219, 219],
            [157, 157, 157],
            [113, 113, 113],
            [70, 70, 70],
            [15, 15, 15],
        ]

    clor = ["#%02x%02x%02x" % (i[0], i[1], i[2]) for i in colorlist]

    # cmap = matplotlib.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],[1., .8, 0.], [1., .4, 0.]])
    cmap = matplotlib.colors.ListedColormap(clor, 'indexed')
    # cmap.set_over((1., 0., 0.))
    # cmap.set_under((0., 0., 1.))

    # bounds = [-1., -.5, 0., .5, 1.]
    bounds = [0.0, 1,  20, 40, 60, 80, 100]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds



def drawmap(rat_fog1, rat_cloud1, outname, titlename, cloudflag = False):
    rat_fog = rat_fog1.copy()
    rat_cloud = rat_cloud1.copy()

    rat_fog[rat_fog==0] = np.nan
    rat_cloud[rat_cloud==0] = np.nan
    cmap2fog, norm2fog, bounds2fog = ColorBar2Fog()
    cmap2cld, norm2cld, bounds2cld = ColorBar2Cld()

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(llcrnrlon=105.0, llcrnrlat=13.0, urcrnrlon=135.0, urcrnrlat=43.0, \
                resolution='i', projection='cyl')

    # m.bluemarble()

    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(13, 43.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
    m.drawmeridians(np.arange(105, 135.01, 5), color='k', linewidth=.5, labels=[1, 0, 0, 1])
    m.fillcontinents(color = '#ddaa66')
    # plt.xlabel('$Longitude$', fontsize=12, labelpad=24)
    # plt.ylabel('$Latitude$', fontsize=12, labelpad=40)

    plt.title(titlename, fontsize=18)

    if cloudflag :
        ffim = m.imshow(rat_cloud[::-1]*100,  cmap=cmap2cld, norm = norm2cld)

    ffim = m.imshow(rat_fog[::-1] * 100, cmap=cmap2fog, norm=norm2fog)

    colorlist = [
        [255, 0, 255], #[237, 28, 36]
        [255, 255, 0],
        [0, 254, 64],
        [0, 255, 255],
        [0, 0, 0],  #ddd#   [255, 0,255]
    ]

    clor = ["#%02x%02x%02x" % (i[0], i[1], i[2]) for i in colorlist]   #  clor = ["#%02x%02x%02x" % (i[0], i[1], i[2]) for i in colorlist]

    contoursol = plt.contour(LON, LAT[::-1], rat_cloud * 100, [10, 30, 50, 70, 100],
                             colors=clor, linewidths=0.8)
    plt.clabel(contoursol, fontsize=8, colors='red', fmt='%.0f')

    ax3 = fig.add_axes([.83, .01, 0.05, 0.05])
    cb3 = plt.colorbar(contoursol, cax=ax3)
    cb3.outline.set_visible(False)


    # 绘制雾的colorbar
    ax1 = fig.add_axes([.1, .02, 0.7, .03])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap2fog,
                                           norm=norm2fog,
                                           # boundaries=[-10] + bounds + [10],
                                           # extend='both', # max, min, both
                                           # Make the length of each extension
                                           # the same as the length of the
                                           # interior colors:
                                           # extendfrac='auto',
                                           ticks=bounds2fog,
                                           spacing='uniform',
                                           orientation='horizontal')
    cb1.set_label('Fog Frequency')

    # 绘制云频次的colorbar
    if cloudflag:
        ax2 = fig.add_axes([.92, .1, 0.03, .75])
        cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap2cld,
                                               norm=norm2cld,
                                               # boundaries=[-10] + bounds + [10],
                                               # extend='both', # max, min, both
                                               # Make the length of each extension
                                               # the same as the length of the
                                               # interior colors:
                                               # extendfrac='auto',
                                               ticks=bounds2cld,
                                               spacing='uniform',
                                               orientation='vertical')
        cb2.set_label('Cloud Frequency')

    # ax3 = fig.add_axes([.08, .02, 0.7, .03])
    # # ax3.colorbar(contoursol)
    # fig.colorbar(contoursol, ax=ax3)
    # # cb1.set_label('Fog Frequency', location='right')

    plt.savefig(outname, dpi=200, bbox_inches='tight')

    plt.close(fig)
    print('draw %s success...' %(outname))


