#coding:utf-8
'''
@project:personal
@author: Lee Bin
@date:2021-02-07
'''
import numpy as np
import fastcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def plot_with_labels(Z, num_clust):
    N = len(Z)
    classes = ['g'] * 50 + ['r'] * 50 + ['c'] * 50

    threshold = Z[-num_clust + 1, 2]
    dg = dendrogram(Z, no_labels = True, color_threshold = threshold)

    color = [classes[k] for k in dg['leaves']]
    b = .1 * Z[-1, 2]
    plt.bar(np.arange(N) * 10, np.ones(N) * b, bottom = -b, width = 10,
            color = color, edgecolor = 'none')
    plt.gca().set_ylim((-b, None))
    plt.show()

if __name__ == '__main__':

    import sys
    if sys.hexversion < 0x03000000:
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    # f = urlopen('http://scipy-cluster.googlecode.com/svn/trunk/hcluster/tests/iris.txt')
    # X = np.loadtxt(f)
    # f.close()
    #
    import random
    X = np.random.rand(10000).reshape(100,100) * 1000

    print(X.shape)
    # Z = fastcluster.linkage(X, method = 'single')
    # plot_with_labels(Z, 2)
    #
    # D = pdist(X, metric = 'cityblock')
    # Z = fastcluster.linkage(D, method = 'weighted')
    # plot_with_labels(Z, 3)


    for method in ["single", "complete", "average", "weighted", "ward", "centroid" , "median"]:
        # D = pdist(X, metric = 'cityblock')
        Z = fastcluster.linkage(X, method = method)
        plot_with_labels(Z, 3)