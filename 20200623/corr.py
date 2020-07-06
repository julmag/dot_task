#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:27:11 2020

@author: jthukral
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns

gc_ac_1 = np.load('gc_rates_1.npz')
gc_ac_1 = gc_ac_1['arr_0']

gc_ac_2 = np.load('gc_rates_2.npz')
gc_ac_2 = gc_ac_2['arr_0']

gc_ac_2_trans = gc_ac_2.transpose()

corr = np.dot(gc_ac_1,gc_ac_2_trans)

corr_norm = np.dot(normalize(gc_ac_1, axis=1, norm='l1'),normalize(gc_ac_2_trans, axis=0, norm='l1'))

# fig, ax = plt.subplots()
# im = ax.imshow(corr)

# # for i in range(len(gc_ac_1)):
# #     for j in range(len(gc_ac_2_trans)):
# #         text = ax.text(j, i, corr[i, j],
# #                        ha="center", va="center", color="w")

# fig.tight_layout()

# plt.show()



ax = sns.heatmap(corr_norm, linewidth=0)
plt.show()

