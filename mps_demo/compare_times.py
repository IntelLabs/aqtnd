# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:04:03 2022

@author: mthibode
"""

import numpy as np
import glob
from copy import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from fractions import Fraction


Dval = 3
listing = glob.glob(f'data/time_data*_L*D{Dval}.npy')
exactdict = defaultdict(list)
approxdict = defaultdict(list)
onesitedict = defaultdict(list)
baddict = defaultdict(list)
for fname in listing:
    x = fname[:-4]
    xl = x.split('_')
    dstr = xl[-1]
    dval = int(dstr[1:])
    lstr = xl[-2]
    lval = int(lstr[1:])
    data = np.load(fname)
    exactval = data[6]/data[2]#
    approxval = data[6]/data[4]#(
    onesiteval = data[6]/data[0]#(
    if data[1] and data[3] and data[5]:
        exactdict[(dval, lval)].append(exactval)
        approxdict[(dval, lval)].append(approxval)
        onesitedict[(dval, lval)].append(onesiteval)
    else:
        baddict[(dval, lval)].append(exactval)

energylisting = glob.glob(f'data/energy_data*_L*D{Dval}.npy')
exact_e_dict = defaultdict(list)
approx_e_dict = defaultdict(list)
for fname in energylisting:
    x = fname[:-4]
    xl = x.split('_')
    dstr = xl[-1]
    dval = int(dstr[1:])
    lstr = xl[-2]
    lval = int(lstr[1:])
    data = np.load(fname)
    exactval = np.real(data[0])
    approxval = np.real(data[1])
    exact_e_dict[(dval, lval)].append(exactval)
    approx_e_dict[(dval, lval)].append(approxval)



# avg and std of each bin
exstatsdict = {}
apstatsdict = {}
exenergystatsdict = {}
apenergystatsdict = {}

for (statd,datad) in [(exstatsdict,exactdict), (apstatsdict, approxdict),
                      (exenergystatsdict,exact_e_dict), (apenergystatsdict, approx_e_dict)]:
    for k in datad.keys():
        statd[k[0]] = []
    for k,v in datad.items():
        m = np.mean(v)
        s = np.std(v)
        d = k[0]
        l = k[1]
        statd[d].append((l,m,s))

keys = exstatsdict.keys()
dims = list(set(keys))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, ax = plt.subplots()
exdata = exactdict[Dval, 40]
apdata = approxdict[Dval, 40]
osdata = onesitedict[Dval, 40]
ax.boxplot([exdata, apdata, osdata], widths = 0.3, sym='')
ax.set_ylabel('Wall-time speedup, X',labelpad=0)
ax.set_xticks([1,2,3], labels=['exact', 'approximate', '1 site'])


fig, ax = plt.subplots()
exedata = exact_e_dict[Dval, 40]
apedata = approx_e_dict[Dval, 40]
ax.boxplot([exedata, apedata], widths = 0.3, sym='')
ax.set_ylabel('Energy error',labelpad=0)
ax.set_xticks([1,2], labels=['exact', 'approximate'])
# plt.figure()
# for (sdict, label) in zip([exstatsdict, apstatsdict], ['exact', 'approx']):
#     for d,v in sdict.items():
#         color = colors.pop(0)
#         vs = sorted(v, key = lambda t: t[0])
#         x,m,s = zip(*vs)
#         plt.errorbar(x, m, yerr=s, c=color, linestyle=None, capsize=5, label=label, marker='x')

# plt.legend()
# plt.xlabel("Length of spin chain (# of tensors)")
# plt.ylabel("% reduction in wall time\nfor GS preparation")
# plt.title("GS prep. is accelerated with entanglement knowledge")
