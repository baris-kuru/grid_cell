#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:00:50 2020

@author: bariskuru
"""

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
    