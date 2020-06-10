#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:37:15 2020

@author: bariskuru
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from skimage.draw import line
import os

''' note that tilted trajectories were cut down to the length of 100 
and prior code is commented out down below '''

#load the data
loaded = np.load('/Users/bariskuru/Desktop/MasterThesis/grids/100_grids_200_arrsize_43_median_spc_2.npz')
all_grids = loaded['all_grids']
mean_grid=loaded['mean_grid'] 
grid_spc=loaded['grid_spc']
grid_ori=loaded['grid_ori'] 
grid_phase=loaded['grid_phase']

#loaded_100 = np.load('/Users/bariskuru/Desktop/grids/400_grids_28to73_size_100.npz')
#all_grids_100 = loaded_100['all_grids']

#parameters
speed_cm = 20
field_size_cm = 100
time_sec = int(field_size_cm/speed_cm)
time_ms = time_sec*1000
start_cm = 75
num_traj = 8
arr_size = all_grids.shape[1]
num_cell = all_grids.shape[2]
dt = (time_sec)/arr_size #for parallel one
dt_ms = dt*1000
size2cm = arr_size/field_size_cm
start_idc = int((size2cm)*(start_cm)-1)

#empty arrays
traj = np.empty((num_cell,arr_size))
traj2 = np.empty((num_cell,arr_size))
par_traj = np.empty((num_cell,arr_size,8))
tilt_traj = np.empty((num_cell,arr_size,8))
# tilt_traj = [] 

#indices for parallel and tilted trajectories
x = np.arange(num_traj-1)
par_idc = np.insert(start_idc-(size2cm*(2**x)), 0, start_idc)
par_idc = par_idc.astype(int)
dev_deg = (2**x)
dev_deg = np.insert(dev_deg,0,0)
dev_deg[7] = 36.999
radian = np.radians(dev_deg)
deviation = np.round(arr_size*np.tan(radian))
deviation = deviation.astype(int)
tilt_idc = start_idc-deviation

#draw the trajectories
start = time.time()
for j in range(num_traj):
    idc = par_idc[j]
    tilt = tilt_idc[j]
    for i in range(num_cell):
        traj[i,:] = profile_line(all_grids[:,:,i], (idc,0), (idc,arr_size-1))
        traj2[i,:] = profile_line(all_grids[:,:,i], (start_idc,0), (tilt, arr_size-1))[:arr_size]
        # sloping trajectories are cut down to the same length of array here
        # traj2.append(profile_line(all_grids[:,:,i], (start_idc,0), (tilt, arr_size-1)))
    par_traj[:,:,j] = traj
    tilt_traj[:,:,j]= traj2
    # tilt_traj.append(traj2)
    # traj2 = []
    
cum_par = np.sum(par_traj, axis=1)*(dt)
cum_tilt = np.sum(tilt_traj, axis=1)*(dt)

    
stop = time.time()
time_min = (stop-start)/60
print(time_min)
print(stop-start)

count = 2 #now variable saved in the grid_func
np.savez(os.path.join('/Users/bariskuru/Desktop/MasterThesis/trajectories', 
                      'trajectories_'+ str(num_cell)+'_cells_'
                      +str(arr_size)+'_arrsize_'+str(count)), 
         all_grids=all_grids,
         tilt_traj=tilt_traj, 
         par_traj=par_traj,
         cum_par = cum_par,
         cum_tilt = cum_tilt,
         num_cell=num_cell, 
         par_idc=par_idc,
         tilt_idc=tilt_idc,
         dev_deg=dev_deg,
         mean_grid=mean_grid, 
         grid_spc=grid_spc, 
         grid_ori=grid_ori, 
         grid_phase=grid_phase,
         count=count)

#
#tilt_traj = profile_line(all_grids[:,:,0], (start_idc,0), (5,arr_size-1))
#plt.figure()
#plt.plot(tilt_traj)
#    
#projection  = all_grids[50, :, :]
#
#
#
#
#
#
#for idc in deviation:
#    plt.close('all')
#    img = all_grids[:, :, 3]
#    rr, cc = line(start_idc, 0, start_idc-idc, arr_size-1)
#    img[rr, cc] = 25
#    plt.imshow(img, cmap='viridis', interpolation='none')
#
#a = profile_line(img, (start_idc,0), (0,arr_size-1), linewidth=1, order=1, mode='constant', cval=25)
#
#
#plt.close('all')
#
#
#plt.plot(projection[:,2])
#plt.figure()
#plt.plot(traj[2,:])
#
#
#start = time.time()
#stop = time.time()
#time_min = (stop-start)/60
#print(time_min)
#print(stop-start)