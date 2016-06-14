#! /usr/bin/env python2
# -*- coding: utf8 -*-
from dataset import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import collections as collections

x1, x2, y, _ = get_test_data()
for i in range(x1.shape[0]):
  if y[i] == 0:
    tmp = x1[i].copy()
    x1[i] = x2[i].copy()
    x2[i] = tmp
    #x1[i], x2[i] = x2[i], x1[i].copy() 
all_x = np.concatenate((x1, x2))
pca = PCA(n_components=2)
pca.fit(all_x)
x1 = pca.transform(x1)
x2 = pca.transform(x2)

print x1, x2

a1 = plt.subplot(221)
good = plt.scatter(x1[:,0], x1[:,1], c='r', alpha=0.5, s=10)
plt.title(u'用户偏好的数据分布')
a2 = plt.subplot(222, sharex=a1, sharey=a1)
bad = plt.scatter(x2[:,0], x2[:,1], c='b', alpha=0.5, s=10)
plt.title(u'用户不偏好据分布')
a3 = plt.subplot(223, sharex=a1, sharey=a1)
xx = [(x1[i], (x1[i] + x2[i]) / 2) for i in range(x1.shape[0])]
lx = collections.LineCollection(xx, color='r', alpha=0.1, linewidth=1)
a3.add_collection(lx)
xx = [(x2[i], (x1[i] + x2[i]) / 2) for i in range(x1.shape[0])]
lx = collections.LineCollection(xx, color='b', alpha=0.1, linewidth=1)
a3.add_collection(lx)
plt.title(u'用户选择情况')
a4 = plt.subplot(224)
xx = [((0, 0), (x2[i] - x1[i])) for  i in range(x1.shape[0])]
lx = collections.LineCollection(xx, color='r', alpha=0.1, linewidth=1)
a4.add_collection(lx)
plt.title(u'用户选择向量分布')
a4.autoscale()
plt.show()

