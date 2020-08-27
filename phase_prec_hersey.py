#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:40:05 2020

@author: bariskuru
"""
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_trajs import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq

spacing = 60
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 20
arr = grid_maker(spacing, 0, [100, 100], arr_size, [field_size_m,field_size_m], max_freq, seed_1=1)



plt.imshow(arr)




sns.reset_orig()
plt.close()
plt.figure()
plt.imshow(arr, cmap='viridis', interpolation=None, extent=[0,100,100,0])
plt.title('2d rate profile of a grid cell \n grid spacing = 50cm')
plt.xlabel('cm')
plt.ylabel('cm')
cbar = plt.colorbar()
cbar.set_label('Hz', rotation=270)
plt.figure()
trans_dist_2d = spacing/2 * np.arccos(3*arr/40-0.5)/2
plt.imshow(trans_dist, cmap='viridis', interpolation=None, extent=[0,100,100,0])
plt.title('Transformed distances from peaks in 2d \n grid spacing = 50cm')
plt.xlabel('cm')
plt.ylabel('cm')
cbar = plt.colorbar()
cbar.set_label('Hz', rotation=270)

sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)

traj = profile_line(arr, (99,0), (99,200-1))
plt.figure()
plt.plot(traj)
plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Rate profile on the trajectory \n grid spacing = 50cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Frequency (Hz)')


traj2 = profile_line(trans_dist_2d, (99,0), (99,200-1))
plt.figure()
plt.plot(traj2)
plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Distance from closest grid peak \n grid spacing = 50cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Distance (cm)')


traj2[np.argmax(traj2)]



plt.close('all')
traj_rate = profile_line(arr, (99,0), (99,200-1))
# trans_norm_2d = np.arccos(3*arr/40-0.5)/2
trans_dist_2d = (np.arccos(((arr*3/(2*max_freq))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
trans_norm_2d = (trans_dist_2d/(spacing/2))/2
traj_loc = profile_line(trans_norm_2d, (99,0), (99,200-1))
traj_rate = profile_line(arr, (99,0), (99,200-1))

# plt.plot(trans_dist_2d)
plt.figure()
plt.plot(traj_loc)

#interpolation
def interp(arr, def_dur_s, new_dt_s):
    arr_len = arr.shape[0]
    t_arr = np.linspace(0, def_dur_s, arr_len)
    new_len = int(def_dur_s/new_dt_s)
    new_t_arr = np.linspace(0, def_dur_s, new_len)
    f = interpolate.interp1d(t_arr, arr)
    interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr


new_dt_s = 0.0001
dur_s = 5
def_dt = dur_s/traj_loc.shape[0]
traj_loc, loc_t_arr = interp(traj_loc, dur_s, new_dt_s)
traj_rate, rate_t_arr = interp(traj_rate, dur_s, new_dt_s)

f=8
T = 1/f
def_time = np.arange(0, dur_s, def_dt)
time_hr = loc_t_arr
# shift = 3*np.pi/2
shift = 0
theta = (np.sin(f*2*np.pi*def_time+shift)+1)/2
theta_hr = (np.sin(f*2*np.pi*time_hr+shift)+1)/2

phase = (2*np.pi*(def_time%T)/T + shift)%(2*np.pi)
phase_hr = (2*np.pi*(time_hr%T)/T + shift)%(2*np.pi)


#infer the direction out of rate of change in the location
direction = np.diff(traj_loc)

loc_hr = np.arange(0, field_size_cm, field_size_cm/traj_loc.shape[0])

#last element is same with the -1 element of diff array
direction = np.append(direction, direction[-1])



threshold = 0.00000
direction[direction < -threshold] = -1
direction[direction > threshold] = 1
direction = -direction

traj_loc_dir = traj_loc*direction
factor = 240/360
firing_phase_dir = 2*np.pi*(traj_loc_dir+0.5)*factor
phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase_hr))
overall_dir = phase_code_dir*traj_rate

plt.figure()
plt.plot(loc_hr, direction)
plt.title('Rate of change at distance from the closest peak \n grid spacing = 50cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Rate of change')


plt.figure()
plt.plot(loc_hr, direction)
plt.title('Direciton of movement in reference to the closest peak \n grid spacing = 50cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Binary Direction')

# direction defines if animal goes into a grid field or goes out of a grid field

plt.plot(loc_hr, traj_loc_dir)
plt.title('Relative location in reference to the closest peak \n grid spacing = 50cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Location/Spacing')

plt.plot(time_hr, firing_phase_dir)
plt.title('Preferred Firing Phase \n grid spacing = 50cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase (radian/pi)')

plt.plot(time_hr, phase_code_dir)
plt.title('Phase Code \n grid spacing = 50cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase code')



plt.figure()

f = plt.figure(figsize=(27,12))
ax = f.add_subplot(211)
plt.plot(rate_t_arr, traj_rate)
# plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Rate profile on the trajectory  \n grid spacing = 50cm')
plt.xlabel('Time (s)')
plt.ylabel('Distance (cm)')

phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase_hr))
ax2 = f.add_subplot(212)
plt.plot(time_hr, phase_code_dir/max(phase_code_dir), label='MPO')
plt.plot(time_hr, theta_hr, label='LFP')
plt.title('Phase Precesion')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Phase')
plt.legend()





#until heee

plt.figure()


time_plt = np.arange(0, dur_s, new_dt_s)



plt.figure()
plt.plot(time_hr, overall_dir)
plt.title('Firing Rate \n grid spacing = 50cm')
plt.xlabel('Time (s)')
plt.ylabel('Hz')

plt.figure()
plt.plot(phase)
plt.plot(theta)
plt.plot(time_hr, 4*theta_hr)

plt.figure()
plt.plot(traj_rate)


###### poisson spikes

dt_s = 0.0001
t_sec = 5
norm_overall = overall_dir
asig = AnalogSignal(norm_overall,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=t_sec*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)


time_bin_size = T
times = np.arange(0, 5+time_bin_size, time_bin_size) 
n_phase_bins = 720
n_time_bins = int(t_sec/time_bin_size)
bins_size_deg = 1

phases = [ [] for _ in range(int((t_sec/time_bin_size))) ]
n = 5000
for i in range(n):
    train = np.array(stg.inhomogeneous_poisson_process(asig))
    # phase_deg.append(train%T/(t_sec*T)*360)
    for j, time in enumerate(times):
        if j == times.shape[0]-1:
            break
        curr_train = train[np.logical_and(train > time, train < times[j+1])]
        if curr_train != []:
            phases[j] += list(curr_train%(T)/(T)*360)
            phases[j] += list(curr_train%(T)/(T)*360+360)
            
# f = plt.figure(figsize=(18,8))
# plt.eventplot(phases, lineoffsets=np.linspace(time_bin_size,t_sec,n_time_bins), linelengths=0.07, linewidths = 1, orientation='vertical')
# # train = np.array(stg.inhomogeneous_poisson_process(asig))
# plt.title('Phases of Poisson Spikes \n' +str(n)+' trials, grid spacing = 50cm')
# plt.xlabel('Time (s)')
# plt.ylabel('Phase (deg)')


counts = np.empty((n_phase_bins, n_time_bins))
                  
for i in range(n_phase_bins):
    for j, phases_in_time in enumerate(phases):
        phases_in_time = np.array(phases_in_time) 
        counts[i][j] = ((bins_size_deg*(i) < phases_in_time) & (phases_in_time < bins_size_deg*(i+1))).sum()

norm_factor = 75
norm_freq = counts*f/n*norm_factor
f2 = plt.figure(figsize=(27,12))
plt.imshow(norm_freq, aspect=1/7, cmap='RdYlBu_r', extent=[0,100,720,0])
plt.ylim((0,720))
#, extent=[0,100,0,720]
plt.title('Mean Firing Frequency ('+str(n)+ ') trials \n phase shift=240deg, spacing='+str(spacing)+'cm, timebin=0.125s, phasebin=1deg')
plt.xlabel('Position (cm)')
plt.ylabel('Theta phase (deg)')
cbar = plt.colorbar()
cbar.set_label('Hz', rotation=270)
# liste = times




f2 = plt.figure(figsize=(27,12))
ax3 = f2.add_subplot(211)
plt.plot(time_hr, overall_dir)
plt.title('Overall Firing Rate \n grid spacing = 50cm')
# plt.xlabel('Time (s)')
plt.ylabel('Hz')

ax4 = f2.add_subplot(212, sharex=ax3)
plt.eventplot(phases, lineoffsets=np.linspace(T,t_sec,40), linelengths=0.07, linewidths = 1, orientation='vertical')
# train = np.array(stg.inhomogeneous_poisson_process(asig))
plt.title('Phases of Poisson Spikes \n' +str(n)+' trials, grid spacing = 50cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')






plt.eventplot(phase_deg, lineoffsets=np.arange(1,n+1,1))
plt.title('Poisson Spikes')
plt.xlabel('Time (s)')
plt.ylabel('Trials w diff seeds')



