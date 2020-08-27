#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:54:03 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from poiss_inp_gen_short_traj import inhom_poiss
from scipy import interpolate
from skimage.measure import profile_line
import os
from pearsonr_ct_bin import ct_a_bin, pearson_r


savedir = os.getcwd()
n_grid = 200 
max_rate = 20
seed = 100
dur_ms = 2000
dur_s = int(dur_ms/1000)
speed_cm = 20
field_size_cm = 100
traj_size_cm = dur_s*speed_cm

grids = grid_population(n_grid, max_rate, seed, arr_size=200)
    
par_trajs = np.array([75, 74.5])
n_traj = par_trajs.shape[0]
par_trajs_pf, dt_s = draw_traj(grids, n_grid, par_trajs, arr_size=200, field_size_cm = field_size_cm, dur_ms=dur_ms, speed_cm=speed_cm)

seed_2s = np.arange(200,205,1)

poiss_spikes = np.empty((n_grid, n_traj))
# binned_counts = np.empty((n_bins, n_grid, n_traj))
poiss_spikes = []


for idx, seed_2 in enumerate(seed_2s):
    curr_spikes = inhom_poiss(par_trajs_pf, n_traj, seed=seed_2, dt_s=dt_s, traj_size_cm=traj_size_cm)
    poiss_spikes.append(curr_spikes)
    
def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    bin_size_sec = bin_size_ms/1000
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.empty((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            counts[i][index] = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            #search and count the number of spikes in the each bin range
    return counts

counts = binned_ct(poiss_spikes[0], 100, time_ms=dur_ms)

# parameters for gc grid generator




np.savez('data_for_perceptron.npz', par_trajs_pf=par_trajs_pf, input_spikes=input_spikes, counts=counts )



        
        
        
        
        
        
        