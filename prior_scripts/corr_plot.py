#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:20:02 2020

@author: bariskuru
"""

import numpy as np
import matplotlib.pyplot as plt
from new_input_gen import inhom_poiss_2
from pearson_corr import pearson_r, mean_corr, binned_corr

loaded_traj = np.load('/Users/bariskuru/Desktop/MasterThesis/trajectories/trajectories_100_cells_200_arrsize_2.npz', allow_pickle=True)
tilt_traj=loaded_traj['tilt_traj']
par_traj=loaded_traj['par_traj']
cum_par = loaded_traj['cum_par']
cum_tilt = loaded_traj['cum_tilt']
num_cell=loaded_traj['num_cell']
par_idc=loaded_traj['par_idc']
tilt_idc=loaded_traj['tilt_idc']
dev_deg=loaded_traj['dev_deg']

spikes_par = inhom_poiss_2(par_traj)
spikes_par_2 = inhom_poiss_2(par_traj)
spikes_ct_1 = binned_corr(spikes_par, 5000)[1].reshape(100,8)
spikes_ct_2 = binned_corr(spikes_par_2, 5000)[1].reshape(100,8)

spikes_tilt = inhom_poiss_2(tilt_traj)
spikes_tilt_2 = inhom_poiss_2(tilt_traj)
spikes_ct_tilt_1 = binned_corr(spikes_tilt, 5000)[1].reshape(100,8)
spikes_ct_tilt_2 = binned_corr(spikes_tilt_2, 5000)[1].reshape(100,8)

par_corr_matrix = np.corrcoef(spikes_ct_1, spikes_ct_2, rowvar=False)
par_corr_matrix = par_corr_matrix[:int(par_corr_matrix.shape[0]/2), int(par_corr_matrix.shape[0]/2):]
par_corr_matrix = np.tril(par_corr_matrix)

tilt_corr_matrix = np.corrcoef(spikes_ct_tilt_1, spikes_ct_tilt_2, rowvar=False)
tilt_corr_matrix = tilt_corr_matrix[:int(tilt_corr_matrix.shape[0]/2), int(tilt_corr_matrix.shape[0]/2):]
tilt_corr_matrix = np.tril(tilt_corr_matrix)

fig, axes = plt.subplots(1,2,figsize=(45,20))
plt.setp((axes[0]) , xticks=np.arange(8), xticklabels=ticks_par_mat, 
         yticks=np.arange(8), yticklabels=ticks_par_mat)
plt.setp((axes[1]) , xticks=np.arange(8), xticklabels=ticks_tilt_mat, 
         yticks=np.arange(8), yticklabels=ticks_tilt_mat)
im1 = axes[0].imshow(par_corr_matrix, aspect='auto', interpolation=None)
im2 = axes[1].imshow(tilt_corr_matrix, aspect='auto', interpolation=None)
axes[0].set_title('Autocorrelation Parallel Traj in 5 sec')
axes[1].set_title('Autocorrelation Sloping Traj in 5 sec')
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])




binned_spi_par_50 = binned_corr(spikes_par, 50)[0]
binned_spi_par_100 = binned_corr(spikes_par, 100)[0]
binned_spi_par_500 = binned_corr(spikes_par, 500)[0]
spi_ct_par_mat = pearson_r(binned_corr(spikes_par, 5000)[1].reshape(100,8))[1] # 5 sec collapsed


spikes_tilt = inhom_poiss_2(tilt_traj)
binned_spi_tilt_50 = binned_corr(spikes_tilt, 50)[0]
binned_spi_tilt_100 = binned_corr(spikes_tilt, 100)[0]
binned_spi_tilt_500 = binned_corr(spikes_tilt, 500)[0]
spi_ct_tilt_mat = pearson_r(binned_corr(spikes_tilt, 5000)[1].reshape(100,8))[1] # 5 sec collapsed

binned_cum_par_50 = binned_corr(par_traj, 50)[0]
binned_cum_par_100 = binned_corr(par_traj, 100)[0]
binned_cum_par_500 = binned_corr(par_traj, 500)[0]
mean_corr_mat_par = mean_corr(par_traj)
cum_par_mat = pearson_r(cum_par)[1] # 5 sec collapsed

binned_cum_tilt_50 = binned_corr(tilt_traj, 50)[0]
binned_cum_tilt_100 = binned_corr(tilt_traj, 100)[0]
binned_cum_tilt_500 = binned_corr(tilt_traj, 500)[0]
mean_corr_mat_tilt = mean_corr(tilt_traj)
cum_tilt_mat = pearson_r(cum_tilt)[1] # 5 sec collapsed



ytick_par = np.array([1111,
                      4311, 4343,
                      5911, 5943, 5959,
                      6711, 6743, 6759, 6767,
                      7111, 7143, 7159, 7167, 7171,
                      7311, 7343, 7359, 7367, 7371, 7373,
                      7411, 7443, 7459, 7467, 7471, 7473, 7474,
                      7511, 7543, 7559, 7567, 7571, 7573, 7574, 7575])

ytick_tilt = ['0000',
              '0100', '0101',
              '0200', '0201', '0202', 
              '0400', '0401', '0402', '0404',
              '0800', '0801', '0802', '0804', '0808',
              '1600', '1601', '1602', '1604', '1608', '1616',
              '3200', '3201', '3202', '3204', '3208', '3216', '3232',
              '3600', '3601', '3602', '3604', '3608', '3616', '3632', '3636']

ticks_par_mat = np.array((par_idc+1)/2, dtype=int)
ticks_tilt_mat = np.flip(dev_deg)

xtick_500 = np.linspace(500,5000, 10, dtype= int)
xtick_100 = np.linspace(200,5000, 25, dtype= int)
xtick_50 = np.linspace(500, 5000, 10, dtype=int)





fig1, axes1 = plt.subplots(2,2,figsize=(45,20))
plt.setp((axes1[0,0],axes1[1,0]) , xticks=np.arange(8), xticklabels=ticks_par_mat, 
         yticks=np.arange(8), yticklabels=ticks_par_mat)
plt.setp((axes1[0,1],axes1[1,1]) , xticks=np.arange(8), xticklabels=ticks_tilt_mat, 
         yticks=np.arange(8), yticklabels=ticks_tilt_mat)
im1 = axes1[0,0].imshow(cum_par_mat, aspect='auto', interpolation=None)
axes1[0,0].set_title('Cumulative Parallel Traj in 5 sec')
fig1.colorbar(im1, ax=axes1[0,0])
im2 = axes1[0,1].imshow(cum_tilt_mat, aspect='auto', interpolation=None)
axes1[0,1].set_title('Cumulative Sloping Traj in 5 sec')
fig1.colorbar(im2, ax=axes1[0,1])
im3 = axes1[1,0].imshow(spi_ct_par_mat, aspect='auto', interpolation=None)
axes1[1,0].set_title('Total Spiking for Parallel Traj in 5 sec')
fig1.colorbar(im3, ax=axes1[1,0])
im4 = axes1[1,1].imshow(spi_ct_tilt_mat, aspect='auto', interpolation=None)
axes1[1,1].set_title('Total Spiking for Sloping Traj in 5 sec')
fig1.colorbar(im4, ax=axes1[1,1])
fig1.savefig('/Users/bariskuru/Desktop/MasterThesis/correlations/5sec_collapsed_matrices.png')


fig2, axes2 = plt.subplots(2,2,figsize=(45,20))
plt.setp((axes2[0,0],axes2[1,0]) , xticks=np.arange(10), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_par)
plt.setp((axes2[0,1],axes2[1,1]) , xticks=np.arange(10), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_tilt)
im21 = axes2[0,0].imshow(binned_cum_par_500, aspect='auto', interpolation=None)
axes2[0,0].set_title('Cumulative Parallel Traj bin=500ms')
axes2[0,0].tick_params(axis='y', labelsize=7 )
fig2.colorbar(im21, ax=axes2[0,0])
im22 = axes2[0,1].imshow(binned_cum_tilt_500, aspect='auto', interpolation=None)
axes2[0,1].set_title('Cumulative Sloping Traj bin=500ms')
axes2[0,1].tick_params(axis='y', labelsize=7 )
fig2.colorbar(im22, ax=axes2[0,1])
im23 = axes2[1,0].imshow(binned_spi_par_500, aspect='auto', interpolation=None)
axes2[1,0].set_title('Spikes Parallel Traj bin=500ms')
axes2[1,0].tick_params(axis='y', labelsize=7 )
fig2.colorbar(im23, ax=axes2[1,0])
im24 = axes2[1,1].imshow(binned_spi_tilt_500, aspect='auto', interpolation=None)
axes2[1,1].set_title('Spikes Sloping Traj bin=500ms')
axes2[1,1].tick_params(axis='y', labelsize=7 )
fig2.colorbar(im24, ax=axes2[1,1])
fig2.savefig('/Users/bariskuru/Desktop/MasterThesis/correlations/correlations_500bin.png')


fig3, axes3 = plt.subplots(2,2,figsize=(45,20))
plt.setp((axes3[0,0],axes3[1,0]) , xticks=np.arange(10, 110, 10), xticklabels=xtick_50, 
         yticks=np.arange(36), yticklabels=ytick_par)
plt.setp((axes3[0,1],axes3[1,1]) , xticks=np.arange(10, 110, 10), xticklabels=xtick_50, 
         yticks=np.arange(36), yticklabels=ytick_tilt)
im21 = axes3[0,0].imshow(binned_cum_par_50, aspect='auto', interpolation=None)
axes3[0,0].set_title('Cumulative Parallel Traj bin=50ms')
axes3[0,0].tick_params(axis='y', labelsize=7 )
fig3.colorbar(im21, ax=axes3[0,0])
im22 = axes3[0,1].imshow(binned_cum_tilt_50, aspect='auto', interpolation=None)
axes3[0,1].set_title('Cumulative Sloping Traj bin=50ms')
axes3[0,1].tick_params(axis='y', labelsize=7 )
fig3.colorbar(im22, ax=axes3[0,1])
im23 = axes3[1,0].imshow(binned_spi_par_50, aspect='auto', interpolation=None)
axes3[1,0].set_title('Spikes Parallel Traj bin=50ms')
axes3[1,0].tick_params(axis='y', labelsize=7 )
fig3.colorbar(im23, ax=axes3[1,0])
im24 = axes3[1,1].imshow(binned_spi_tilt_50, aspect='auto', interpolation=None)
axes3[1,1].set_title('Spikes Sloping Traj bin=50ms')
axes3[1,1].tick_params(axis='y', labelsize=7 )
fig3.colorbar(im24, ax=axes3[1,1])
fig3.savefig('/Users/bariskuru/Desktop/MasterThesis/correlations/correlations_50bin.png')


fig4, axes4 = plt.subplots(2,2,figsize=(45,20))
plt.setp((axes4[0,0],axes4[1,0]) , xticks=np.arange(5, 55, 5), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_par)
plt.setp((axes4[0,1],axes4[1,1]) , xticks=np.arange(5, 55, 5), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_tilt)
im21 = axes4[0,0].imshow(binned_cum_par_100, aspect='auto', interpolation=None)
axes4[0,0].set_title('Cumulative Parallel Traj bin=100ms')
axes4[0,0].tick_params(axis='y', labelsize=7 )
fig4.colorbar(im21, ax=axes4[0,0])
im22 = axes4[0,1].imshow(binned_cum_tilt_100, aspect='auto', interpolation=None)
axes4[0,1].set_title('Cumulative Sloping Traj bin=100ms')
axes4[0,1].tick_params(axis='y', labelsize=7 )
fig4.colorbar(im22, ax=axes4[0,1])
im23 = axes4[1,0].imshow(binned_spi_par_100, aspect='auto', interpolation=None)
axes4[1,0].set_title('Spikes Parallel Traj bin=100ms')
axes4[1,0].tick_params(axis='y', labelsize=7 )
fig4.colorbar(im23, ax=axes4[1,0])
im24 = axes4[1,1].imshow(binned_spi_tilt_100, aspect='auto', interpolation=None)
axes4[1,1].set_title('Spikes Sloping Traj bin=100ms')
axes4[1,1].tick_params(axis='y', labelsize=7 )
fig4.colorbar(im24, ax=axes4[1,1])
fig4.savefig('/Users/bariskuru/Desktop/MasterThesis/correlations/correlations_100bin.png')



fig1 = plt.figure(figsize=(45,20))
par_cum_img_500 = plt.imshow(binned_cum_par_500, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(10), xtick_500)
plt.colorbar(par_cum_img_500).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n Cumulative  bin = 500 ms')

fig2 = plt.figure(figsize=(45,20))
par_cum_img_500 = plt.imshow(binned_spi_par_500, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(10), xtick_500)
plt.colorbar(par_cum_img_500).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n Spikes  bin = 500 ms')


fig11 = plt.figure(figsize=(45,20))
par_cum_img_100 = plt.imshow(binned_cum_par_50, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(0, 100, 2), xtick_50)
plt.colorbar(par_cum_img_100).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n bin = 100 ms')

fig3 = plt.figure(figsize=(45,20))
par_cum_img_100 = plt.imshow(binned_cum_par_100, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(0, 50, 2), xtick_100)
plt.colorbar(par_cum_img_100).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n bin = 100 ms')

fig4 = plt.figure(figsize=(45,20))
par_cum_img_100 = plt.imshow(binned_spi_par_100, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(0, 50, 2), xtick_100)
plt.colorbar(par_cum_img_100).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n bin = 100 ms')

fig4 = plt.figure(figsize=(45,20))
par_cum_img_100 = plt.imshow(binned_cum_par_100, aspect='auto', interpolation=None)
plt.ytick_pars(np.arange(36), ytick_par)
plt.xticks(np.arange(0, 50, 2), xtick_100)
plt.colorbar(par_cum_img_100).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation between parallel trajectories \n bin = 100 ms')

fig5 = plt.figure(figsize=(45,20))
# img1 = plt.imshow(mat_cum_par)
img1 = plt.imshow(mean_mat)
# labels = list(['75','74','73','71','67','59', '43', '11'])

plt.ytick_pars(np.arange(8), ticks_par_mat)
plt.xticks(np.arange(8), ticks_par_mat)
plt.colorbar(img).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation of Integrated Firing for 100 grid cells \n Parallel Trajectories collapsed in 5 seconds')
# plt.figure()
# img2 = plt.imshow(mat_cum_tilt)

fig6 = plt.figure(figsize=(45,20))
img3 = plt.imshow(mat_spike_par)
plt.ytick_pars(np.arange(8), labels)
plt.xticks(np.arange(8), labels)
plt.colorbar(img).set_label('pearson r', labelpad=15, rotation=270)
plt.title('Correlation of Total Number of Spikes (Poisson) for 100 grid cells \n Parallel Trajectories collapsed in 5 seconds')
plt.ytick_pars(np.arange(8), labels)
plt.figure()
plt.imshow(mat_spike_tilt)



fig1, axes1 = plt.subplots(2,2,figsize=(45,20))
plt.setp(axes1, xticks=np.arange(10), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_par)


axes1[0,0].imshow()
axes1[0,0].set_xlabel()
axes1[0,0].set_ylabel()
axes1[0,0].set_title()
axes1[0,1].imshow()
axes1[0,1].set_xlabel()
axes1[0,1].set_ylabel()
axes1[0,1].set_title()
axes1[0,1].legend()
axes1[1,0].imshow()
axes1[1,0].set_xlabel()
axes1[1,0].set_ylabel()
axes1[1,0].set_title()
axes1[1,1].imshow()
axes1[1,1].set_xlabel()
axes1[1,1].set_ylabel()
axes1[1,1].set_title()
axes1[1,1].legend()

fig2, axes2 = plt.subplots(2,2,figsize=(45,20))
axes1[0,0].imshow()
axes1[0,1].imshow()
axes1[1,0].imshow()
axes1[1,1].imshow()

fig3, axes3 = plt.subplots(2,2,figsize=(45,20))
axes1[0,0].imshow()
axes1[0,1].imshow()
axes1[1,0].imshow()
axes1[1,1].imshow()

fig4, axes4 = plt.subplots(2,2,figsize=(45,20))
axes1[0,0].imshow()
axes1[0,1].imshow()
axes1[1,0].imshow()
axes1[1,1].imshow()

#template for 4 subplots




fig1.colorbar(im1,im2,im3,im4)
fig1.colorbar(im1
             fig1.colorbar(im1

fig1, axes1 = plt.subplots(2,2,figsize=(45,20))
plt.setp(axes1, xticks=np.arange(10), xticklabels=xtick_500, 
         yticks=np.arange(36), yticklabels=ytick_par)
axes1[0,0].imshow()
axes1[0,0].tick_params(axis='y', labelsize=7)
axes1[0,0].set_xlabel()
axes1[0,0].set_ylabel()
axes1[0,0].set_title()
axes1[0,1].imshow()
axes1[0,1].tick_params(axis='y', labelsize=7)
axes1[0,1].set_xlabel()
axes1[0,1].set_ylabel()
axes1[0,1].set_title()
axes1[0,1].legend()
axes1[1,0].imshow()
axes1[1,0].tick_params(axis='y', labelsize=7)
axes1[1,0].set_xlabel()
axes1[1,0].set_ylabel()
axes1[1,0].set_title()
axes1[1,1].imshow()
axes1[1,1].tick_params(axis='y', labelsize=7)
axes1[1,1].set_xlabel()
axes1[1,1].set_ylabel()
axes1[1,1].set_title()
axes1[1,1].legend()

