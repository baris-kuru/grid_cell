#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:47:29 2020

@author: bariskuru
"""

import numpy as np
from scipy.stats import skewnorm
from skimage.measure import profile_line


def grid_maker(spacing, orientation, pos_peak, arr_size, sizexy, max_rate):
    #define the params from input here, scale the resulting array for maxrate and sperate the xy for size and shift
    arr_size = arr_size
    x, y = pos_peak
    pos_peak = np.array([x,y])
    max_rate = max_rate
    lambda_spacing = spacing*(arr_size/100) #100 required for conversion, they have probably used 100*100 matrix in 
    k = (4*np.pi)/(lambda_spacing*np.sqrt(3))
    degrees = orientation
    theta = np.pi*(degrees/180)
    meterx, metery = sizexy
    arrx = meterx*arr_size # *arr_size for defining the 2d array size
    arry = metery*arr_size
    dims = np.array([arrx,arry])
    arr = np.ones(dims)
    
    #implementation of grid function
    # 3 ks for 3 cos gratings with different angles
    k1 = ((k/np.sqrt(2))*np.array((np.cos(theta+(np.pi)/12) + np.sin(theta+(np.pi)/12),
          np.cos(theta+(np.pi)/12) - np.sin(theta+(np.pi)/12)))).reshape(2,)
    k2 = ((k/np.sqrt(2))*np.array((np.cos(theta+(5*np.pi)/12) + np.sin(theta+(5*np.pi)/12),
          np.cos(theta+(5*np.pi)/12) - np.sin(theta+(5*np.pi)/12)))).reshape(2,)
    k3 = ((k/np.sqrt(2))*np.array((np.cos(theta+(9*np.pi)/12) + np.sin(theta+(9*np.pi)/12),
          np.cos(theta+(9*np.pi)/12) - np.sin(theta+(9*np.pi)/12)))).reshape(2,)
    #.reshape is only need when function is in the loop(shape somehow becomes (2,1) otherwise normal shape is already (2,)
    for i in range(dims[0]):
        for j in range(dims[1]):
            curr_pos = np.array([i,j]-pos_peak)
            arr[i,j] = (np.cos(np.dot(k1, curr_pos))+
               np.cos(np.dot(k2, curr_pos))+ np.cos(np.dot(k3, curr_pos)))/3
    arr = max_rate*2/3*(arr+1/2)   # arr is the resulting 2d grid out of 3 gratings      
    return arr

def grid_population(n_grid, max_rate, seed, arr_size=200):
    # skewed normal distribution for grid_spc
    np.random.seed(seed)
    median_spc = 43
    spc_max = 100
    skewness = 6  #Negative values are left skewed, positive values are right skewed.
    grid_spc = skewnorm.rvs(a = skewness,loc=spc_max, size=n_grid)  #Skewnorm function
    grid_spc = grid_spc - min(grid_spc)      #Shift the set so the minimum value is equal to zero.
    grid_spc = grid_spc / max(grid_spc)      #Standadize all the vlues between 0 and 1. 
    grid_spc = grid_spc * spc_max         #Multiply the standardized values by the maximum value.
    grid_spc = grid_spc + (median_spc - np.median(grid_spc))
    
    grid_ori = np.random.randint(0, high=60, size=[n_grid,1]) #uniform dist btw 0-60 degrees
    grid_phase = np.random.randint(0, high=(arr_size-1), size=[n_grid,2]) #uniform dist grid phase
    
    # create a 3d array with grids for n_grid
    all_grids = np.zeros((arr_size, arr_size, n_grid))#empty array
    for i in range(n_grid):
        x = grid_phase[i][0]
        y = grid_phase[i][1]
        arr = grid_maker(grid_spc[i], grid_ori[i], [x, y], arr_size, [1,1], max_rate)
        all_grids[:, :, i] = arr
    
    return all_grids, grid_spc

def draw_traj(all_grids ,n_grid, par_trajs, arr_size=200, field_size_cm = 100, dur_ms=5000, speed_cm=20):
    "Activation profile of cells out of trajectories walked by a mouse" 
    
    all_grids = all_grids
    size2cm = int(arr_size/field_size_cm)
    dur_s = int(dur_ms/1000)
    traj_len_cm = dur_s*speed_cm
    traj_len_dp = traj_len_cm*size2cm
    dt_s = (dur_s)/traj_len_cm #for parallel one
    par_idc_cm = par_trajs
    par_idc = par_idc_cm*size2cm-1
    n_traj = par_idc.shape[0]
    #empty arrays
    traj = np.empty((n_grid,traj_len_dp))
    par_traj = np.empty((n_grid,traj_len_dp,n_traj))
    
    #draw the trajectories
    for j in range(n_traj):
        idc = par_idc[j]
        for i in range(n_grid):
            traj[i,:] = profile_line(all_grids[:,:,i], (idc,0), (idc,traj_len_dp-1))
        par_traj[:,:,j] = traj
    # cum_par = np.sum(par_traj, axis=1)*(dt_s)
    
    return par_traj, dt_s

