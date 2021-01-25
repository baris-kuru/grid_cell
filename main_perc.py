#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:05:14 2020

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
from rate_n_phase_code_gra import phase_code, overall_spike_ct
import time
import copy
from neuron import h, gui  # gui necessary for some parameters to h namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim



start = time.time()
# Where to search for nrnmech.dll file. Must be adjusted for your machine. For pyDentate
dll_files = [("C:\\Users\\DanielM\\Repos\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              "C:\\Users\\daniel\\Repos\\nrnmech.dll",
              ("C:\\Users\\Holger\\danielm\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              ("C:\\Users\\Daniel\\repos\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              ("/home/baris/Python/mechs_7-6_linux/"
                "x86_64/.libs/libnrnmech.so")]

for x in dll_files:
    if os.path.isfile(x):
        dll_dir = x
print("DLL loaded from: " + dll_dir)
h.nrn_load_dll(dll_dir)

#Parameters for the grid cell poisson input generation
savedir = os.getcwd()+'/asd'
n_grid = 200 
n_gra = 2000
max_rate = 20
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = dur_ms/1000
speed_cm = 20
field_size_cm = 100
traj_size_cm = int(dur_s*speed_cm)
inp_len = n_bin*n_gra

#Parameters for perceptron
lr_grid = 5e-4
lr_gra = 5e-3 #was 5e-3 and good for 500ms, and for 2000ms 5e-4 was set
n_iter = 10000
th = 0.2

pp_weight=9e-4

#Seeds
grid_seeds = 510
poiss_seeds = np.arange(200,205,1)
perc_seeds = grid_seeds-100

n_poiss = poiss_seeds.shape[0]
n_network = 1 #perc_seeds.shape[0]

#similar & distinct trajectories
sim_traj = np.array([75, 74.5])
diff_traj = np.array([75, 60])
n_traj = sim_traj.shape[0]

#Intialize zeros arrays&lists to fill with data
sample_size = 2*poiss_seeds.shape[0]
n_sampleset = 1 #perc_seeds.shape[0]


#labels, output for training the network, 5 for each trajectory
for i in poiss_seeds:
    a = np.tile([1, 0], (len(poiss_seeds),1))
    b = np.tile([0, 1], (len(poiss_seeds),1))
    labels = np.vstack((a,b))
labels = torch.FloatTensor(labels) 
out_len = labels.shape[1]




#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, labels, n_iter=1000, lr=1e-4):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    track_loss = []
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    for i in range(n_iter):
        out = net(train_data)
        loss = torch.sqrt(loss_fn(out, labels))
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
    
        # Update weights
        optimizer.step()
    
        # Store current value of loss
        track_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar
        
        # Track progress
        if (i + 1) % (n_iter // 5) == 0:
          print(f'iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}')

    return track_loss, out



def perceptron(sim_traj_cts, diff_traj_cts, phases_sim, phases_diff, perc_seed, lr):
    
    #threshold crossing points
    th_cross = np.zeros(6)
    #change rate code to mean of non zeros where it is nonzero
    cts_for_phase_sim = copy.deepcopy(gra_sim_traj_cts)
    cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(cts_for_phase_sim[cts_for_phase_sim!=0]) #was 1
    cts_for_phase_diff = copy.deepcopy(gra_diff_traj_cts)
    cts_for_phase_diff[cts_for_phase_diff!=0]=np.mean(cts_for_phase_diff[cts_for_phase_diff!=0])
    
    #rate code with constant 45 deg phase
    phase_of_rate_code = np.pi/4
    rate_y_sim = gra_sim_traj_cts*np.sin(phase_of_rate_code)
    rate_x_sim = gra_sim_traj_cts*np.cos(phase_of_rate_code)
    rate_sim =  np.concatenate((rate_y_sim, rate_x_sim), axis=1)
    rate_y_diff = gra_diff_traj_cts*np.sin(phase_of_rate_code)
    rate_x_diff = gra_diff_traj_cts*np.cos(phase_of_rate_code)
    rate_diff =  np.concatenate((rate_y_diff, rate_x_diff), axis=1)
    
    #phase code with phase and mean rate 
    phase_y_sim = cts_for_phase_sim*np.sin(gra_phases_sim)
    phase_x_sim = cts_for_phase_sim*np.cos(gra_phases_sim)
    phase_sim =  np.concatenate((phase_y_sim, phase_x_sim), axis=1)
    phase_y_diff = cts_for_phase_diff*np.sin(gra_phases_diff)
    phase_x_diff = cts_for_phase_diff*np.cos(gra_phases_diff)
    phase_diff =  np.concatenate((phase_y_diff, phase_x_diff), axis=1)
    #complex code with rate and phase
    complex_sim_y = gra_sim_traj_cts*np.sin(gra_phases_sim)
    complex_sim_x = gra_sim_traj_cts*np.cos(gra_phases_sim)
    complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
    complex_diff_y = gra_diff_traj_cts*np.sin(gra_phases_diff)
    complex_diff_x = gra_diff_traj_cts*np.cos(gra_phases_diff)
    complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)

    #Into tensor
    rate_sim_t = torch.FloatTensor(rate_sim)
    rate_diff_t = torch.FloatTensor(rate_diff)
    phase_sim_t = torch.FloatTensor(phase_sim)
    phase_diff_t = torch.FloatTensor(phase_diff)
    complex_sim_t = torch.FloatTensor(complex_sim)
    complex_diff_t = torch.FloatTensor(complex_diff)

    #initate the network with diff types of inputs and plot the change in loss
    #rate code
    torch.manual_seed(perc_seed)
    net_rate_sim = Net(inp_len*2, out_len)
    rate_train_loss_sim, rate_out_sim = train_net(net_rate_sim, rate_sim_t, labels, n_iter=n_iter, lr=lr)
    th_cross[0] = np.argmax(np.array(rate_train_loss_sim) < th)
    torch.manual_seed(perc_seed)
    net_rate_diff = Net(inp_len*2, out_len)
    rate_train_loss_diff, rate_out_diff = train_net(net_rate_diff, rate_diff_t, labels, n_iter=n_iter, lr=lr)
    th_cross[1] = np.argmax(np.array(rate_train_loss_diff) < th)
    #phase code        
    torch.manual_seed(perc_seed)
    net_phase_sim = Net(inp_len*2, out_len)
    phase_train_loss_sim, out_sim = train_net(net_phase_sim, phase_sim_t, labels, n_iter=n_iter, lr=lr)
    th_cross[2] = np.argmax(np.array(phase_train_loss_sim) < th)
    torch.manual_seed(perc_seed)
    net_phase_diff = Net(inp_len*2, out_len)
    phase_train_loss_diff, out_diff = train_net(net_phase_diff, phase_diff_t, labels, n_iter=n_iter, lr=lr)
    th_cross[3] = np.argmax(np.array(phase_train_loss_diff) < th)
    #complex code
    torch.manual_seed(perc_seed)
    net_complex_sim = Net(inp_len*2, out_len)
    complex_train_loss_sim, complex_out_sim = train_net(net_complex_sim, complex_sim_t, labels, n_iter=n_iter, lr=lr)
    th_cross[4] = np.argmax(np.array(complex_train_loss_sim) < th)
    torch.manual_seed(perc_seed)
    net_complex_diff = Net(inp_len*2, out_len)
    complex_train_loss_diff, complex_out_diff = train_net(net_complex_diff, complex_diff_t, labels, n_iter=n_iter, lr=lr)
    th_cross[5] = np.argmax(np.array(complex_train_loss_diff) < th)
    
    rate_code = np.stack((rate_sim, rate_diff), axis=2)
    phase_code = np.stack((phase_sim, phase_diff), axis=2)
    complex_code = np.stack((complex_sim, complex_diff), axis=2)

    return rate_code, phase_code, complex_code, th_cross



# in case you wanna use diff perc seeds in one console, put this part into a loop

#Input generation
#Rate trajs with phase info; oscillations implemented in the rate profile
grid_phases_sim,  gra_phases_sim, grid_spikes_sim, gra_spikes_sim, rate_trajs_sim, dt_s, theta_phase_sim, phase_code_dir_sim, overall_dir_sim = phase_code(sim_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight)
grid_phases_diff, gra_phases_diff, grid_spikes_diff, gra_spikes_diff, rate_trajs_diff, dt_s, theta_phase_diff, phase_code_dir_diff, overall_dir_diff = phase_code(diff_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight)
#grid and granule spike counts \ rate codes
grid_sim_traj_cts, gra_sim_traj_cts = overall_spike_ct(grid_spikes_sim, gra_spikes_sim, dur_ms, poiss_seeds, n_traj=n_traj)
grid_diff_traj_cts, gra_diff_traj_cts = overall_spike_ct(grid_spikes_diff, gra_spikes_diff, dur_ms, poiss_seeds, n_traj=n_traj)
 
grid_rate_code, grid_phase_code, grid_complex_code, grid_th_cross = perceptron(grid_sim_traj_cts, grid_diff_traj_cts, grid_phases_sim, grid_phases_diff, perc_seeds, lr_grid)
gra_rate_code, gra_phase_code, gra_complex_code, gra_th_cross = perceptron(gra_sim_traj_cts, gra_diff_traj_cts, gra_phases_sim, gra_phases_diff, perc_seeds, lr_gra)



save_dir = '/home/baris/repo/perceptron_results/'
fname = 'rate_n_phase_perceptron_net-seed'+str(perc_seeds)+'_'+str(dur_ms)+'ms'

np.savez(save_dir+fname, 
         grid_phases_sim = grid_phases_sim,
         grid_phases_diff = grid_phases_diff,
         grid_spikes_sim = grid_spikes_sim,
         grid_spikes_diff = grid_spikes_diff,
         grid_sim_traj_cts = grid_sim_traj_cts,
         grid_diff_traj_cts = grid_diff_traj_cts,
         
         grid_rate_code = grid_rate_code,
         grid_phase_code = grid_phase_code,
         grid_complex_code = grid_complex_code,
         grid_th_cross = grid_th_cross,
         
         gra_phases_sim = gra_phases_sim,
         gra_phases_diff = gra_phases_diff,
         gra_spikes_sim = gra_spikes_sim,
         gra_spikes_diff = gra_spikes_diff,
         gra_sim_traj_cts = gra_sim_traj_cts,
         gra_diff_traj_cts = gra_diff_traj_cts,
         
         gra_rate_code = gra_rate_code,
         gra_phase_code = gra_phase_code,
         gra_complex_code = gra_complex_code,
         gra_th_cross = gra_th_cross,
         
         n_grid = n_grid, 
         pp_weight = pp_weight,
         max_rate = max_rate,
         dur_ms = dur_ms,
         bin_size = bin_size,
         n_bin = n_bin,
         dur_s = dur_s,
         speed_cm = speed_cm,
         field_size_cm = field_size_cm,
         traj_size_cm = traj_size_cm,
         inp_len = inp_len,
         lr_grid = lr_grid,
         lr_gra = lr_gra,
         n_iter = n_iter,
         sample_size = sample_size,
         n_sampleset = n_sampleset,
         labels = labels,
         sim_traj = sim_traj,
         diff_traj = diff_traj,
         grid_seeds = grid_seeds,
         poiss_seeds = poiss_seeds,
         perc_seeds = perc_seeds)

stop = time.time()
print('time, sec, min, hour  ')
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour)


