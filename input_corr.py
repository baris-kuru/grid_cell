#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:36:14 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import os
from grid_poiss_input_gen import inhom_poiss
from grid_trajs import grid_population, draw_traj
from pearsonr_ct_bin import ct_a_bin, pearson_r
import matplotlib.pyplot as plt
import matplotlib.font_manager #for linux

savedir = os.getcwd()
input_scale = 1000

seed_1 = 101 #seed_1 for granule cell generation
seed_2_1 = 200 #seed_2 inh poisson input generation
seed_2_2 = 201
seed_3 = seed_1+50 #seed_3 for network generation & simulation
seeds=np.array([seed_1, seed_2_1, seed_2_2])

#number of cells
n_grid = 200 
n_granule = 2000
n_mossy = 60
n_basket = 24
n_hipp = 24

# parameters for gc grid generator
arr_size = 200 # arr_size is the side length of the square field as array length
field_size_cm = 100 #size of the field in real life 
speed_cm = 20 # speed of the virtual mouse
dur_ms = (field_size_cm/speed_cm)*1000
dur_ms = 5000
# par_trajs = np.array([75,74.5,74,73.5,73,72.5,72,71.5,71,70.5,70,69,68,67,66,65,64,63,62,61,60,58,56,54,52,48,43,37,32,26,20,13])
par_trajs = np.arange(100, 2, -2)
n_traj = par_trajs.shape[0]
max_rate = 20

def difference_matrix(a):
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()
diff_matrix = difference_matrix(par_trajs)
iu = np.triu_indices(diff_matrix.shape[0], k=1)
diff_matrix = diff_matrix[iu]


np.random.seed(seed_1) #seed_1 for granule cell generation
grids = grid_population(n_grid, max_rate, seed_1)[0]
par_traj, dur_ms, dt_s = draw_traj(grids, n_grid, par_trajs)


# generate temporal patterns out of grid cell act profiles as an input for pyDentate
input_100 = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed_2_1)
input_101 = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed_2_2)


counts_100 = ct_a_bin(input_100, 2000, 2500)
counts_100_1 = ct_a_bin(input_100, 1000, 1500)
counts_100_2 = ct_a_bin(input_100, 3000, 3500)
counts_101 = ct_a_bin(input_101, 2000, 2500)
counts_101_1 = ct_a_bin(input_101, 1000, 1500)
counts_101_2 = ct_a_bin(input_101, 3000, 3500)


diff_trajs = pearson_r(counts_100, counts_100)[1]


# poiss_noise = pearson_r(counts_100, counts_101)[0]

               

'Plotting'
sns.set(context='paper',style='whitegrid',palette='colorblind',font='Arial',font_scale=1.5,color_codes=True)
plt.figure()
sns.scatterplot(diff_matrix, diff_trajs)
plt.title('Scatterplot of Input Correlations between ' +str(n_traj)+' Parallel Trajectories Ranging in 2-100 cm')
plt.xlabel('Distance between trajectories (cm)')
plt.ylabel('Rin')
parameters = ('bin size = 500 ms      max rate = '+str(max_rate)+
              '       seed1='+str(seed_1)+', seed2='+str(seed_2_1)+', seed3='+str(seed_3))
plt.annotate(parameters, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)






