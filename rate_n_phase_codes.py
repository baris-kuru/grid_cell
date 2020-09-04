#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:41:50 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
import os

savedir = os.getcwd()
n_grid = 200 
max_rate = 20
seed = 100
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = int(dur_ms/1000)
speed_cm = 20
field_size_cm = 100
traj_size_cm = dur_s*speed_cm
def_dt_s = 0.025
new_dt_s = 0.002
dt_s = new_dt_s


plt.close('all')


def rate2dist(grids, spacings, max_rate):
    grid_dist = np.empty((grids.shape[0], grids.shape[1], grids.shape[2]))
    for i in range(grids.shape[2]):
        grid = grids[:,:,i]
        spacing = spacings[i]
        trans_dist_2d = (np.arccos(((grid*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
        grid_dist[:,:,i] = (trans_dist_2d/(spacing/2))/2
        
    return grid_dist

#interpolation
def interp(arr, dur_s, dt_s, new_dt_s):
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, dur_s, arr_len)
    if new_dt_s != dt_s: #if dt given is different than default_dt_s(0.025), then interpolate
        new_len = int(dur_s/new_dt_s)
        new_t_arr = np.linspace(0, dur_s, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr




def inhom_poiss(arr, n_traj, seed_2=0, dt_s=0.025):
    #length of the trajectory that mouse went
    np.random.seed(seed_2)
    n_cells = arr.shape[0]
    spi_arr = np.empty((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(seed+grid_idc)
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur_s*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig)
            spi_arr[grid_idc, i] = np.array(curr_train.times*1000) #time conv to ms
    return spi_arr



def mean_phase(spikes, T, n_phase_bins, n_time_bins, times):
    rad = n_phase_bins/360*2*np.pi
    phases = rad*np.ones((spikes.shape[0], n_time_bins, spikes.shape[1]))
    spikes_s = spikes/1000
    for idx, val in np.ndenumerate(spikes_s):
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = val[np.logical_and(val > time, val < times[j+1])]
            if curr_train.size != 0:
                phases[idx[0],j,idx[1]] = np.mean(curr_train%(T)/(T)*rad)
            
    return phases


#Count the number of spikes in bins 

def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.empty((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            counts[i][index] = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            #search and count the number of spikes in the each bin range
    return counts


def spike_ct(trajs_pf):

    seed_2s = np.arange(200,205,1)
    n_traj = 2
    poiss_spikes = []
    counts_1 = np.empty((len(seed_2s), n_bin*n_grid))
    counts_2 = np.empty((len(seed_2s), n_bin*n_grid))
    for idx, seed_2 in enumerate(seed_2s):
        curr_spikes = inhom_poiss(trajs_pf, n_traj, seed_2=seed_2, dt_s=dt_s)
        poiss_spikes.append(curr_spikes)
        counts_1[idx, :] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_2[idx,:] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,1].flatten()
    counts = np.vstack((counts_1, counts_2))
    return counts



def phase_code(trajs, seed_1, seed_2s, f=10, shift_deg=240):

    T = 1/f
    time_bin_size = T
    times = np.arange(0, dur_s+time_bin_size, time_bin_size) 
    n_phase_bins = 360
    n_time_bins = int(dur_s/time_bin_size)
    bins_size_deg = 1
    
    grids, spacings = grid_population(n_grid, max_rate, seed=seed_1, arr_size=arr_size)
    sim_traj = np.array(trajs)
    grid_dist = rate2dist(grids, spacings, max_rate)
    dist_trajs, dt_s = draw_traj(grid_dist, n_grid, sim_traj, dur_ms=dur_ms)
    dist_trajs, dist_t_arr = interp(dist_trajs, dur_s, def_dt_s, new_dt_s)
    rate_trajs, rate_dt_s = draw_traj(grids, n_grid, sim_traj, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = interp(rate_trajs, dur_s, def_dt_s, new_dt_s)
    
    
    dt_s = new_dt_s
    # theta = (np.sin(f*2*np.pi*dist_t_arr)+1)/2
    one_theta_phase = (2*np.pi*(dist_t_arr%T)/T)%(2*np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis,:], 200, axis=0)
    theta_phase =  np.repeat(theta_phase[:,:,np.newaxis], 2, axis=2)
    
    #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction = -direction
    
    traj_dist_dir = dist_trajs*direction
    factor = shift_deg/360 #change the phase shift from 360 degrees to 240 degrees
    firing_phase_dir = 2*np.pi*(traj_dist_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-theta_phase))
    factor = 75
    overall_dir = phase_code_dir*rate_trajs*factor
    
    phases_1 = np.empty((len(seed_2s), n_bin*n_grid))
    phases_2 = np.empty((len(seed_2s), n_bin*n_grid))
    for idx, seed_2 in enumerate(seed_2s):
        spikes = inhom_poiss(overall_dir, 2, dt_s=dt_s, seed_2=seed_2)
        curr_phases = mean_phase(spikes, T, n_phase_bins, n_time_bins, times)
        curr_phases = curr_phases.reshape(-1, curr_phases.shape[-1]).T
        phases_1[idx, :] = curr_phases[0]
        phases_2[idx:] = curr_phases[1]
    phases = np.vstack((phases_1, phases_2))
    return phases, rate_trajs, dt_s
    
    
