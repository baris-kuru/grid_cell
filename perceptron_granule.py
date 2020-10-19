#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:28:00 2020

@author: bariskuru
"""


import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
from rate_n_phase_codes import phase_code, spike_ct, gra_spike_to_phase
import time

#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, train_labels, n_iter=1000, lr=1e-4):
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


#Parametersfor the grid cell poisson input generation
savedir = os.getcwd()
n_grid = 200 
n_gra = 2000
max_rate = 20
dur_ms = 200
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = dur_ms/1000
speed_cm = 20
field_size_cm = 100
traj_size_cm = int(dur_s*speed_cm)
inp_len = n_bin*n_gra

#Parameters for perceptron
lr = 1e-1
n_iter = 10000

#Initialize the figures
fig1, ax1 = plt.subplots()
ax1.set_title('Rate Code Perceptron Loss '+str(dur_ms)+'ms \n multip torch seeds, learning rate = '+str(lr))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('RMSE Loss')

fig2, ax2 = plt.subplots()
ax2.set_title('Phase Code Perceptron Loss '+str(dur_ms)+'ms \n multip torch seeds, learning rate = '+str(lr))
ax2.set_xlabel('Epochs')
ax2.set_ylabel('RMSE Loss')

fig3, ax3 = plt.subplots()
ax3.set_title('Phase*Rate Code Perceptron Loss '+str(dur_ms)+'ms \n multip torch seeds, learning rate = '+str(lr))
ax3.set_xlabel('Epochs')
ax3.set_ylabel('RMSE Loss')

fig4, ax4 = plt.subplots()
ax4.set_title('Complex Phase&Rate Code Perceptron Loss '+str(dur_ms)+'ms \n multip torch seeds, learning rate = '+str(lr))
ax4.set_xlabel('Epochs')
ax4.set_ylabel('RMSE Loss')

#Seeds: seed1 for grids, seed2 for poiss spikes, seed4 for network
seed_1s = np.arange(100,120,1)
seed_2s = np.arange(200,205,1)
seed_4s = np.arange(0,20,1)

#Intialize empty arrays&lists to fill with data
sample_size = 2*seed_2s.shape[0]
n_sampleset = seed_4s.shape[0]
rate_code_sim = np.empty((sample_size, inp_len, n_sampleset))
rate_code_diff = np.empty((sample_size, inp_len, n_sampleset))
phase_code_sim = np.empty((sample_size, inp_len, n_sampleset))
phase_code_diff = np.empty((sample_size, inp_len, n_sampleset))
rate_phase_code_sim = np.empty((sample_size, inp_len, n_sampleset))
rate_phase_code_diff = np.empty((sample_size, inp_len, n_sampleset))


complex_code_sim = np.empty((sample_size, 2*inp_len, n_sampleset))
complex_code_diff = np.empty((sample_size, 2*inp_len, n_sampleset))

rate_th_cross_sim = []
rate_th_cross_diff = []
phase_th_cross_sim = []
phase_th_cross_diff = []
rate_phase_th_cross_sim = []
rate_phase_th_cross_diff = []

complex_th_cross_sim = []
complex_th_cross_diff = []

#labels, output for training the network, 5 for each trajectory
labels = torch.FloatTensor([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],
                            [0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]) 

out_len = labels.shape[1]

#main loop
# generate the grid data with different seeds
# put them into phase and rate code funstions and collect the data for perceptron
# generate the network with different seeds and plot the change in loss
start = time.time()
for idx, seed_4 in enumerate(seed_4s):
    #similar & distinct trajectories
    sim_traj = np.array([75, 74.5])
    diff_traj = np.array([75, 60])
    
    #Input generation
    _, rate_trajs_sim, dt_s = phase_code(sim_traj, dur_ms, seed_1s[idx], seed_2s)
    _, rate_trajs_diff, dt_s = phase_code(diff_traj, dur_ms, seed_1s[idx], seed_2s)

    grid_sim_traj_cts, gra_sim_traj_cts, grid_sim_spikes, gra_sim_spikes = spike_ct(rate_trajs_sim, dur_ms)
    grid_diff_traj_cts, gra_diff_traj_cts, grid_diff_spikes, gra_diff_spikes = spike_ct(rate_trajs_diff, dur_ms)

    #granule cell phase code generation
    
    phases_sim = gra_spike_to_phase (gra_sim_spikes, seed_2s)
    phases_diff = gra_spike_to_phase (gra_diff_spikes, seed_2s)
    
    rate_phase_sim = gra_sim_traj_cts*phases_sim
    rate_phase_diff = gra_diff_traj_cts*phases_diff
    
    
    complex_sim_y = gra_sim_traj_cts*np.sin(phases_sim)
    complex_sim_x = gra_sim_traj_cts*np.cos(phases_sim)
    complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
    complex_diff_y = gra_diff_traj_cts*np.sin(phases_diff)
    complex_diff_x = gra_diff_traj_cts*np.cos(phases_diff)
    complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)
    


    
    #Normalization
    gra_sim_traj_cts = gra_sim_traj_cts/np.amax(gra_sim_traj_cts)
    gra_diff_traj_cts = gra_diff_traj_cts/np.amax(gra_diff_traj_cts)
    phases_sim = phases_sim/np.amax(phases_sim)
    phases_diff = phases_diff/np.amax(phases_diff)
    rate_phase_sim = rate_phase_sim/np.amax(rate_phase_sim)
    rate_phase_diff = rate_phase_diff/np.amax(rate_phase_diff)
    
    complex_sim = complex_sim/np.amax(complex_sim)
    complex_diff = complex_diff/np.amax(complex_diff)
    
    #fill arrays to save the data
    rate_code_sim[:,:,idx] = gra_sim_traj_cts
    rate_code_diff[:,:,idx] = gra_diff_traj_cts
    phase_code_sim[:,:,idx] = phases_sim
    phase_code_diff[:,:,idx] = phases_diff
    rate_phase_code_sim[:,:,idx] = rate_phase_sim
    rate_phase_code_diff[:,:,idx] = rate_phase_diff

    complex_code_sim[:,:,idx] = complex_sim
    complex_code_diff[:,:,idx] = complex_diff
    
    print('data done!')

    #Into tensor
    rate_sim = torch.FloatTensor(gra_sim_traj_cts)
    rate_diff = torch.FloatTensor(gra_diff_traj_cts)
    
    phase_sim = torch.FloatTensor(phases_sim)
    phase_diff = torch.FloatTensor(phases_diff)

    rate_phase_sim = torch.FloatTensor(rate_phase_sim)
    rate_phase_diff = torch.FloatTensor(rate_phase_diff)
    
    complex_sim = torch.FloatTensor(complex_sim)
    complex_diff = torch.FloatTensor(complex_diff)

    #initate the network with diff types of inputs and plot the change in loss
    
    #rate code
    torch.manual_seed(seed_4)
    net_rate_sim = Net(inp_len, out_len)
    rate_train_loss_sim, rate_out_sim = train_net(net_rate_sim, rate_sim, labels, n_iter=n_iter, lr=lr)
    rate_th_cross_sim.append(np.argmax(np.array(rate_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax1.plot(rate_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax1.plot(rate_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_rate_diff = Net(inp_len, out_len)
    rate_train_loss_diff, rate_out_diff = train_net(net_rate_diff, rate_diff, labels, n_iter=n_iter, lr=lr)
    rate_th_cross_diff.append(np.argmax(np.array(rate_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax1.plot(rate_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax1.plot(rate_train_loss_diff, 'r-')
        
    #phase code        
    torch.manual_seed(seed_4)
    net_phase_sim = Net(inp_len, out_len)
    phase_train_loss_sim, out_sim = train_net(net_phase_sim, phase_sim, labels, n_iter=n_iter, lr=lr)
    phase_th_cross_sim.append(np.argmax(np.array(phase_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax2.plot(phase_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax2.plot(phase_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_phase_diff = Net(inp_len, out_len)
    phase_train_loss_diff, out_diff = train_net(net_phase_diff, phase_diff, labels, n_iter=n_iter, lr=lr)
    phase_th_cross_diff.append(np.argmax(np.array(phase_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax2.plot(phase_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax2.plot(phase_train_loss_diff, 'r-')
    
    #rate*phase
    torch.manual_seed(seed_4)
    net_rate_phase_sim = Net(inp_len, out_len)
    rate_phase_train_loss_sim, rate_phase_out_sim = train_net(net_rate_phase_sim, rate_phase_sim, labels, n_iter=n_iter, lr=lr)
    rate_phase_th_cross_sim.append(np.argmax(np.array(rate_phase_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax3.plot(rate_phase_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax3.plot(rate_phase_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_rate_phase_diff = Net(inp_len, out_len)
    rate_phase_train_loss_diff, rate_phase_out_diff = train_net(net_rate_phase_diff, rate_phase_diff, labels, n_iter=n_iter, lr=lr)
    rate_phase_th_cross_diff.append(np.argmax(np.array(rate_phase_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax3.plot(rate_phase_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax3.plot(rate_phase_train_loss_diff, 'r-')
        
    #complex 
    torch.manual_seed(seed_4)
    net_complex_sim = Net(inp_len, out_len)
    complex_train_loss_sim, complex_out_sim = train_net(net_complex_sim, complex_sim, labels, n_iter=n_iter, lr=lr)
    complex_th_cross_sim.append(np.argmax(np.array(complex_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax4.plot(complex_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax4.plot(complex_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_complex_diff = Net(inp_len, out_len)
    complex_train_loss_diff, complex_out_diff = train_net(net_complex_diff, complex_diff, labels, n_iter=n_iter, lr=lr)
    complex_th_cross_diff.append(np.argmax(np.array(complex_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax4.plot(complex_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax4.plot(complex_train_loss_diff, 'r-')
        
        
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
#add threshold line at 0.2
ax1.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')
ax2.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')
ax3.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')
ax4.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')

# plt.annotate(str(th_cross_sim)+'\n'+str(th_cross_diff), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)
fname = 'granule_rate_n_phase_perceptron_norm_'+str(dur_ms)+'ms_'+str(n_iter)+'_iter_'+str(lr)+'_lr'
np.savez(fname, 
         rate_code_sim = rate_code_sim,
         rate_code_diff = rate_code_diff,
         phase_code_sim = phase_code_sim,
         phase_code_diff = phase_code_diff,
         rate_phase_code_sim = rate_phase_code_sim,
         rate_phase_code_diff = rate_phase_code_diff,
         complex_code_sim = complex_code_sim,
         complex_code_diff = complex_code_diff,
        
         rate_th_cross_sim=rate_th_cross_sim, 
         rate_th_cross_diff=rate_th_cross_diff,
         phase_th_cross_sim=phase_th_cross_sim, 
         phase_th_cross_diff=phase_th_cross_diff,
         rate_phase_th_cross_diff=rate_phase_th_cross_diff, 
         rate_phase_th_cross_sim=rate_phase_th_cross_sim,
         complex_th_cross_diff=complex_th_cross_diff, 
         complex_th_cross_sim=complex_th_cross_sim,
        
         n_grid = n_grid, 
         max_rate = max_rate,
         dur_ms = dur_ms,
         bin_size = bin_size,
         n_bin = n_bin,
         dur_s = dur_s,
         speed_cm = speed_cm,
         field_size_cm = field_size_cm,
         traj_size_cm = traj_size_cm,
         inp_len = inp_len,
         lr = lr,
         n_iter = n_iter,
         sample_size = sample_size,
         n_sampleset = n_sampleset,
         labels = labels,
         sim_traj = sim_traj,
         diff_traj = diff_traj,
         seed_1s = seed_1s,
         seed_2s = seed_2s,
         seed_4s = seed_4s)


stop = time.time()
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour)

