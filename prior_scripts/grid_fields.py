#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:50:46 2020

@author: bariskuru
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

plt.close('all')

dims = np.array([1000,1000])
init = np.ones(dims)
disposition = np.random.randint(-20, high=20, size=[400,2])
spacing = np.random.randint(6, high=10, size=[400,1])
interval = np.linspace(0,1,dims[1])

all_grids = np.zeros((dims[0],dims[1], 400))

for i in range(400):
    arr = init
    w = 4*np.pi*spacing[i]
    sin_wave = np.cos(w*interval)
    arr = arr*sin_wave
    arr_60 = ndimage.rotate(arr, 60, reshape=False, mode='wrap')
    arr_120 = ndimage.rotate(arr, 120, reshape=False, mode='wrap')
    grid_arr = 2/3*((arr + arr_60 + arr_120)/3 + 1/2)
    grid_arr = np.roll(grid_arr, [disposition[i][0], disposition[i][1]], axis=(0,1))
    all_grids[:, :, i] = grid_arr
    
    

