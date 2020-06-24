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
import matplotlib.pyplot as plt

savedir = os.getcwd()
input_scale = 1000

seed_1 = 100 #seed_1 for granule cell generation
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
par_trajs = np.array([75,74.5,74,73,72,70,67,64,60,56,52,48,43,37,32,26,20,13])
n_traj = par_trajs.shape[0]
max_rate = 20

# def difference_matrix(a):
#     x = np.reshape(a, (len(a), 1))
#     return x - x.transpose()
# diff_matrix = difference_matrix(par_trajs)
# iu = np.triu_indices(diff_matrix.shape[0], k=1)
# diff_matrix = np.sort(diff_matrix[iu])
#     # corr arr is the values vectorized 
#     diag_low = corr_mat[iu]
# plt.imshow(diff_matrix)
# plt.colorbar()



np.random.seed(seed_1) #seed_1 for granule cell generation
grids = grid_population(n_grid, max_rate, seed_1)[0]
par_traj, par_idc_cm, dur_ms, dt_s = draw_traj(grids, n_grid, par_trajs)


# generate temporal patterns out of grid cell act profiles as an input for pyDentate
input_100 = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed_2_1)
input_101 = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed_2_2)

def ct_a_bin(arr, bin_start, bin_end):
    counts = np.empty((arr.shape[0], arr.shape[1]))
    for index, value in np.ndenumerate(arr):
        # print(index)
        counts[index] = ((value > bin_start) & (value< bin_end)).sum()
    return counts

counts_100 = ct_a_bin(input_100, 2000, 2500)
counts_100_1 = ct_a_bin(input_100, 1000, 1500)
counts_100_2 = ct_a_bin(input_100, 3000, 3500)
counts_101 = ct_a_bin(input_101, 2000, 2500)
counts_101_1 = ct_a_bin(input_101, 1000, 1500)
counts_101_2 = ct_a_bin(input_101, 3000, 3500)

def pearson_r(x,y):
    #corr mat is doubled in each axis since it is 2d*2d
    corr_mat = np.corrcoef(x, y, rowvar=False) 
    #slice out the 1 of 4 identical mat
    corr_mat = corr_mat[int(corr_mat.shape[0]/2):, :int(corr_mat.shape[0]/2)] 
    # indices in upper triangle
    iu =np.triu_indices(int(corr_mat.shape[0]), k=1)
    # corr arr is the values vectorized 
    diag_low = corr_mat[iu]
    diag = corr_mat.diagonal()
    return diag, diag_low
diff_trajs = pearson_r(counts_100, counts_100)[1]
diff_trajs_3 = np.concatenate((pearson_r(counts_100, counts_100)[1], 
              pearson_r(counts_100_1, counts_100_1)[1],
              pearson_r(counts_100_2, counts_100_2)[1]),  axis=None)

poiss_noise = pearson_r(counts_100, counts_101)[0]
poiss_noise_3 = np.concatenate((pearson_r(counts_100, counts_101)[0],
               pearson_r(counts_100_1, counts_101_1)[0],
               pearson_r(counts_100_2, counts_101_2)[0]), axis=None)
               

'Plotting'
sns.set(context='paper',style='whitegrid',palette='colorblind',font='Arial',font_scale=1.5,color_codes=True)
plt.figure()
hist_bin = 0.02
sns.distplot(diff_trajs, np.arange(0, 1+hist_bin, hist_bin), rug=True)
sns.distplot(poiss_noise, np.arange(0, 1+hist_bin, hist_bin), rug=True)
plt.legend(('diff trajs w same poiss seed', 'same trajs w diff poisson seeds'))  
plt.title('Distributions of Input Correlations in 500ms Bin \n(75, 74.5, 74, 73.5)')
plt.xlabel('Rin')
plt.ylabel('Count')






