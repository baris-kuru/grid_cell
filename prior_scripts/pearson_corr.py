#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:15:05 2020

@author: bariskuru
"""

'''
spacings 
38.8(0.5), 48.4, 65, 98.4 & 8cm variation

orientations
15,30,45,60 3 degree variation
in Neher 2015 paper

'''
import numpy as np


def pearson_r(x):
    corr_mat = np.corrcoef(x, x, rowvar=False) 
    #corr mat is doubled in each axis since it is 2d*2d
    corr_mat = corr_mat[:int(corr_mat.shape[0]/2), :int(corr_mat.shape[0]/2)] 
    #slice out the 1 of 4 identical mat
    iu =np.triu_indices(int(corr_mat.shape[0])) 
    # indices in upper triangle
    corr_arr = corr_mat[iu]
    # corr arr is the values vectorized 
    corr_mat = np.tril(corr_mat)
    # corr_mat is values in lower triangle(better look, symmetrical with upper) in a matrix
    return corr_arr, corr_mat

def mean_corr(arr, dt_ms=25, time_ms=5000):
    mean_arr = np.mean(arr, axis=1)
    mean_corr_mat = pearson_r(mean_arr)[1]
    return mean_corr_mat


def binned_corr(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    bin_size_sec = bin_size_ms/1000
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    
    if np.ndim(arr) == 3: #if input is 
    # if array here, then output of trajectory.py
    # structure is [#cells, len trajs, #trajs]
        bin_split = np.split(arr, n_bins, axis=1) 
        # split array into #bin pieces for the length of traj(len trajs(200)/#bins), 
        #and we have #bins times (#cells(100),#trajs(8)) arrays to compare in pearson_r
        n_traj = arr.shape[2]
        binned_corr = np.zeros((sum(range(n_traj+1)), n_bins))
        #create an array with dims of total # traj comparison (36 for 8 traj) and n_bins
        for i in range(n_bins):
            # loop through bins
            binned = np.mean(bin_split[i], axis=1)*bin_size_sec
            # calculate the avarage activation in each bin and multiply by dt
            # like avarage number of spikes in each bin calculated by cumulative graph
            # then you have area under the graph for each bin, now array dims for each bin is 100*8
            binned_corr[:,i] = np.flip(pearson_r(binned)[0])
            #put the array into pearson_r and flip
            counts = []
    elif np.ndim(arr) == 2:
        # if ndim 2 then it is poisson spikes
        n_traj = arr.shape[1]
        counts = np.empty((n_bins, n_cells, n_traj))
        binned_corr = np.zeros((sum(range(n_traj+1)), n_bins))
        #create an array with dims of total # traj comparison (36 for 8 traj) and n_bins
        for i in range(n_bins):
            for index, value in np.ndenumerate(arr):
                counts[i][index] = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
                #search and count the number of spikes in the each bin range
            binned_corr[:,i] = np.flip(pearson_r(counts[i])[0])
            #put the array into pearson_r and flip
    else:
        raise Exception("Dimensions of the array are wrong")
    
    # return binned_corr
    return binned_corr, counts
