#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:29:29 2020

@author: bariskuru
"""
import time
import numpy as np
import matplotlib.pyplot as plt
loaded_traj = np.load('/Users/bariskuru/Desktop/trajectories/trajectories_400_cells_200_arrsize.npz', allow_pickle=True)
tilt_traj=loaded_traj['tilt_traj']
par_traj=loaded_traj['par_traj']
num_cell=loaded_traj['num_cell']
par_idc=loaded_traj['par_idc']
tilt_idc=loaded_traj['tilt_idc']
dev_deg=loaded_traj['dev_deg']
all_grids = loaded_traj['all_grids']
mean_grid=loaded_traj['mean_grid'] 
grid_spc=loaded_traj['grid_spc']
grid_ori=loaded_traj['grid_ori'] 
grid_phase=loaded_traj['grid_phase']

field_size_cm = 100
speed_cm = 20
traj_start = 75
par_arr_length = par_traj[0].shape[0]
par_dur_sec = field_size_cm/speed_cm
par_dur_sec = np.linspace(0,par_dur_sec,par_arr_length)

plt.close('all')
for grid_idc in [0,5,1,6]:
    # grid_idc = np.array([7,9,0,3])
    spacing = np.around(grid_spc[grid_idc], decimals=1)
    orientation = grid_ori[grid_idc]
    timelist_sec = []
    fig = str(spacing)
    fig, axes = plt.subplots(2,2,figsize=(45,20))
    for i in range(dev_deg.shape[0]):
        arr_length = tilt_traj[i][0].shape[0]
        dist_cm = arr_length/2
        dur_sec = dist_cm/speed_cm
        dur_sec = np.linspace(0,dur_sec,arr_length)
        timelist_sec.append(dur_sec)
        axes[1,1].plot(dur_sec, tilt_traj[i][grid_idc], label=str(dev_deg[i])+' deg')
        axes[0,1].plot(par_dur_sec, par_traj[grid_idc][:,i], label=str((par_idc[i]+1)/2)+' cm')
        axes[0,0].plot([0,99],[((par_idc[i]/2)+0.5),((par_idc[i]/2)+0.5)])
        axes[1,0].plot([0,100],[75, ((tilt_idc[i]/2)+0.5)])
    img = axes[0,0].imshow(all_grids[:,:,grid_idc],cmap='jet', interpolation='none', extent=[0, 100, 100, 0])
    cbar = plt.colorbar(img, ax=axes[0,0])
    cbar.set_label('Hz', labelpad=15, rotation=270)
    img2 = axes[1,0].imshow(all_grids[:,:,grid_idc],cmap='jet', interpolation='none', extent=[0, 100, 100, 0])
    cbar2 = plt.colorbar(img2, ax=axes[1,0])
    cbar2.set_label('Hz', labelpad=15, rotation=270)
    
    axes[0,0].set_xlabel('distance (cm)')
    axes[0,0].set_ylabel('distance (cm)')
    axes[0,0].set_title('Grid field w '+ str(spacing) + ' cm spacing  | Parallel')
    
    axes[1,0].set_xlabel('distance (cm)')
    axes[1,0].set_ylabel('distance (cm)')
    axes[1,0].set_title('Grid field w '+ str(spacing) + ' cm spacing | Tilted')
    
    axes[0,1].set_xlabel('Time (sec)')
    axes[0,1].set_ylabel('Firing rate (Hz)')
    axes[0,1].set_title('Rate profile vs Time | Parallel')
    axes[0,1].legend()
    
    axes[1,1].set_xlabel('Time (sec) ')
    axes[1,1].set_ylabel('Firing rate (Hz)')
    axes[1,1].set_title('Rate profile vs Time  | Tilted')
    axes[1,1].legend()
    
    # fig.savefig('/Users/bariskuru/Desktop/MasterThesis/trajectories/trajectory_('+str(spacing)+').png')

'''
YAPILACAKLAR

 - subfigures
 - draw lines on grids and put aside rate profile of lines with same colors
'''