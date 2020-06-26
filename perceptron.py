#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:14:57 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import os
from grid_poiss_input_gen import inhom_poiss
from grid_trajs import grid_population, draw_traj
import matplotlib.pyplot as plt
from pearsonr_ct_bin import ct_a_bin, pearson_r


savedir = os.getcwd()
input_scale = 1000

seed_1 = 100 #seed_1 for granule cell generation
seed_2 = 200 #seed_2 inh poisson input generation
seed_3 = seed_1+50 #seed_3 for network generation & simulation
seed_4 = 50 #seed for initializing parameters in perceptron
seeds = np.array([seed_1, seed_2, seed_3, seed_4])
#number of cells
n_grid = 200 
n_granule = 2000
n_mossy = 60
n_basket = 24
n_hipp = 24

# parameters for gc grid generator
par_trajs = np.arange(75,60,-0.5)
n_traj = par_trajs.shape[0]
max_rate = 20


np.random.seed(seed_1) #seed_1 for granule cell generation
grids = grid_population(n_grid, max_rate, seed_1)[0]
par_trajs_pf = draw_traj(grids, n_grid, par_trajs)[0]
input_spikes = inhom_poiss(par_trajs_pf, n_traj, dt_s=0.0001, seed=seed_2)
counts = ct_a_bin(input_spikes, [2000,2500])

np.savez('data_for_perceptron.npz', par_trajs_pf=par_trajs_pf, input_spikes=input_spikes, counts=counts )

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(200,3)
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data, nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform(m.bias.data)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
    
   
net = Net()
net.cuda()

net = Net()
print(net)
print(list(net.parameters()))
input = Variable(torch.randn(1,1,1), requires_grad=True)
print(input)

out = net(input)
print(out)


# def criterion(out, label):
#     return (label - out)**2



optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)


data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

outp = np.zeros(30,3)
for i in range(n_traj):
    print(i)
    if i < 10:
        outp[i,0] = 1
    elif i > 19:
        outp[i,2] = 1
    else:
        outp[i,1] = 1

torch.cuda.manual_seed(seed_4)
for epoch in range(10):
    for i, data_inp in enumerate(data):
        X = data_inp
        Y = outp[i,:]
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True).cuda(), Variable(torch.FloatTensor([Y]), requires_grad=False).cuda()
        optimizer.zero_grad()
        outputs = net(X)
        criterion = nn.MSELoss()
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))










