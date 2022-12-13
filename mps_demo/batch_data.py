# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:16:53 2022

@author: mthibode
"""

import numpy as np
import h5py
from glob import glob

L = 40 
D = 2
getL = str(L)

flist = glob(f'data/cos_random/run_data*L{getL}*D{D}.npy')

with h5py.File(f'datasets/mps_cos_heis_data_L{getL}_D{D}.h5' , 'w') as fh5:
    dataset = fh5.create_dataset('heis-bd-data', (len(flist), L, 4))
    
    for j in range(len(flist)):
        thisdata = np.load(flist[j])
        dataset[j] = thisdata
    
    

