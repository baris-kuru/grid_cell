#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:04:12 2020

@author: bariskuru
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
from rate_n_phase_codes import phase_code, spike_ct

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
max_rate = 20
seed = 100
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = int(dur_ms/1000)
speed_cm = 20
field_size_cm = 100
traj_size_cm = dur_s*speed_cm





lr = 1e-4
n_iter = 1000

fig1, ax1 = plt.subplots()
ax1.set_title('Perceptron Learning Rate Code for Similar Trajectories\n multip torch seeds, learning rate = '+str(lr))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('RMSE Loss')

fig2, ax2 = plt.subplots()
ax2.set_title('Perceptron Learning Phase Code for Similar Trajectories\n multip torch seeds, learning rate = '+str(lr))
ax2.set_xlabel('Epochs')
ax2.set_ylabel('RMSE Loss')

seed_1s = np.arange(100,120,1)
seed_2s = np.arange(200,205,1)
seed_4s = np.arange(0,20,1)

rate_th_cross_sim = []
rate_th_cross_diff = []
phase_th_cross_sim = []
phase_th_cross_diff = []
for idx, seed_4 in enumerate(seed_4s):
    
    sim_traj = np.array([75, 74.5])
    diff_traj = np.array([75, 60])
    
    
    phases_sim, rate_trajs_sim, dt_s = phase_code(sim_traj, seed_1s[idx], seed_2s)
    phases_diff, rate_trajs_diff, dt_s = phase_code(diff_traj, seed_1s[idx], seed_2s)

    sim_traj_cts = spike_ct(rate_trajs_sim)
    diff_traj_cts = spike_ct(rate_trajs_diff)
    
    print('data done!')

    rate_sim = torch.FloatTensor(sim_traj_cts)
    rate_diff = torch.FloatTensor(diff_traj_cts)
    
    phase_sim = torch.FloatTensor(phases_sim)
    phase_diff = torch.FloatTensor(phases_diff)

    labels = torch.FloatTensor([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],
                                [0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]) 
    torch.manual_seed(seed_4)
    net_rate_sim = Net(4000,2)
    rate_train_loss_sim, rate_out_sim = train_net(net_rate_sim, rate_sim, labels, n_iter=n_iter, lr=lr)
    rate_th_cross_sim.append(np.argmax(np.array(rate_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax1.plot(rate_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax1.plot(rate_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_rate_diff = Net(4000,2)
    rate_train_loss_diff, rate_out_diff = train_net(net_rate_diff, rate_diff, labels, n_iter=n_iter, lr=lr)
    rate_th_cross_diff.append(np.argmax(np.array(rate_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax1.plot(rate_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax1.plot(rate_train_loss_diff, 'r-')
            
    torch.manual_seed(seed_4)
    net_phase_sim = Net(4000,2)
    phase_train_loss_sim, out_sim = train_net(net_phase_sim, phase_sim, labels, n_iter=n_iter, lr=lr)
    phase_th_cross_sim.append(np.argmax(np.array(phase_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax2.plot(phase_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax2.plot(phase_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_phase_diff = Net(4000,2)
    phase_train_loss_diff, out_diff = train_net(net_phase_diff, phase_diff, labels, n_iter=n_iter, lr=lr)
    phase_th_cross_diff.append(np.argmax(np.array(phase_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax2.plot(phase_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax2.plot(phase_train_loss_diff, 'r-')

plt.legend()

# plt.annotate(str(th_cross_sim)+'\n'+str(th_cross_diff), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)


