#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:17:08 2020

@author: bariskuru
"""

import numpy as np
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
import matplotlib.pyplot as plt
import time

loaded_traj = np.load('/Users/bariskuru/Desktop/trajectories/trajectories_100_cells_200_arrsize.npz', allow_pickle=True)
tilt_traj=loaded_traj['tilt_traj']
par_traj=loaded_traj['par_traj']
cum_par = loaded_traj['cum_par']
cum_tilt = loaded_traj['cum_tilt']
num_cell=loaded_traj['num_cell']
par_idc=loaded_traj['par_idc']
tilt_idc=loaded_traj['tilt_idc']
dev_deg=loaded_traj['dev_deg']
all_grids = loaded_traj['all_grids']
mean_grid=loaded_traj['mean_grid'] 
grid_spc=loaded_traj['grid_spc']
grid_ori=loaded_traj['grid_ori'] 
grid_phase=loaded_traj['grid_phase']

arr_size = all_grids.shape[1]
field_size_cm = 100
factor_size2cm = arr_size/field_size_cm
speed_cm = 20
arr_length = par_traj[0].shape[0]
dur = field_size_cm/speed_cm
time_sec = np.linspace(0,dur, arr_length)


plt.close('all')
def inhom_poiss():
    par_spike_trains = []
    tilt_spike_trains = []
    n_inputs = par_traj.shape[0]
    par_spike_ct = np.zeros((n_inputs, dev_deg.shape[0]))
    tilt_spike_ct = np.zeros((n_inputs, dev_deg.shape[0]))
    par_spikes = []
    tilt_spikes = []
    for grid_idc in range(n_inputs):
        for i in range(dev_deg.shape[0]):
            rate_profile = par_traj[grid_idc,:,i]
            rate_profile_tilt = tilt_traj[grid_idc,:,i]
            # rate_profile_tilt = tilt_traj[i, grid_idc]
            # tilt_dur = (len(rate_profile_tilt)/factor_size2cm)/speed_cm
            # line above for diff length of tilt traj
            asig_par = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur*pq.s,
                                    sampling_period=0.025*pq.s,
                                    sampling_interval=0.025*pq.s)
            asig_tilt = AnalogSignal(rate_profile_tilt,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur*pq.s,
                                    sampling_period=0.025*pq.s,
                                    sampling_interval=0.025*pq.s)
        
            curr_train_par = stg.inhomogeneous_poisson_process(asig_par)
            par_spike_trains.append(curr_train_par)
            par_spike_ct[grid_idc, i] = curr_train_par.shape[0]
            curr_train_tilt = stg.inhomogeneous_poisson_process(asig_tilt)
            tilt_spike_trains.append(curr_train_tilt)
            tilt_spike_ct[grid_idc, i] = curr_train_tilt.shape[0]
            
        par_array = np.array([np.around(np.array(x.times)*1000, decimals=1) for x in par_spike_trains])
        par_spikes.append(par_array)
        tilt_array = np.array([np.around(np.array(x.times)*1000, decimals=1) for x in tilt_spike_trains])
        tilt_spikes.append(tilt_array)
    return par_spikes, tilt_spikes, par_spike_ct, tilt_spike_ct

start = time.time()

par_spikes, tilt_spikes, par_spike_ct, tilt_spike_ct = inhom_poiss()

stop = time.time()
time_min = (stop-start)/60
print(time_min)
print(stop-start)

# plt.eventplot(par_spikes[3], linelengths=0.5, color='black')
# labels = list(['75cm','74cm','73cm','71cm','67cm', '59cm', '43cm', '11cm'])
# plt.yticks(np.arange(8), labels)

# plt.figure()
# plt.eventplot(tilt_spikes[3], linelengths=0.5, color='black')
# labels = list(['0 deg','1 deg','2 deg','4 deg','8 deg','16 deg', '32 deg', '36 deg'])
# plt.yticks(np.arange(8), labels)

# def time_stamps_to_signal(time_stamps, dt_signal, t_start, t_stop):
#     """Convert an array of timestamps to a signal where 0 is absence and 1 is
#     presence of spikes
#     """
#     # Construct a zero array with size corresponding to desired output signal
#     sig = np.zeros((np.shape(time_stamps)[0],int((t_stop-t_start)/dt_signal)))
#     # Find the indices where spikes occured according to time_stamps
#     time_idc = []
#     for x in time_stamps:
#         curr_idc = []
#         curr_idc.append((x-t_start)/ dt_signal)
#         time_idc.append(curr_idc)
    
#     # Set the spike indices to 1
#     for sig_idx, idc in enumerate(time_idc):
#         sig[sig_idx,np.array(idc,dtype=np.int)] = 1

#     return sig


# time_stamps_to_signal(par_time_sec, 0.025, 0, 5)

# plt.plot(rate_profile_as_asig)
# plt.figure()
# plt.plot(rate_profile)

# plt.plot(curr_train)

# plt.eventplot(curr_train, linelengths=0.75, color='black')



