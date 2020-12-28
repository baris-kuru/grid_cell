# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:07:19 2020

@author: Daniel
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

data = np.load("granule_rate_n_phase_perceptron_2000ms_net-seeds_410-419.npz", allow_pickle=True)


rate_code_sim = data['rate_code_sim'][0][1,:,0]
phase_code_sim = data['phase_code_sim'][0][1,:,0]
complex_code_sim = data['complex_code_sim'][0][1,:,0]
mean_rate = np.mean(rate_code_sim)
mean_phase = np.mean(np.abs(phase_code_sim))
mean_complex = np.mean(np.abs(complex_code_sim))


fig, ax = plt.subplots()
ax.plot(data['rate_code_sim'][0][1,:,0], alpha=0.3)
ax.plot(data['phase_code_sim'][0][1,:,0], alpha=0.3)
ax.plot(data['complex_code_sim'][0][1,:,0], alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))




fig, ax = plt.subplots(1,3)
for x in [0,1]:
    ax[0].plot(data['rate_code_diff'][0][x,:,0], alpha=0.5)
    ax[1].plot(data['phase_code_diff'][0][x,:,0], alpha=0.5)
    ax[2].plot(data['complex_code_diff'][0][x,:,0], alpha=0.5)

ax[0].set_title("Rate Code")
ax[1].set_title("Phase Code")
ax[2].set_title("Complex Code")

# Calculate mean pearson R
codes = ['rate_code_diff', 'phase_code_diff', 'complex_code_diff',
         'rate_code_sim', 'phase_code_sim', 'complex_code_sim']
corr_dict = {}
for code in codes:
    corr_matrices = []
    for grid in range(data[code].shape[0]):
        corr_matrix = []
        for traj_l in range(data[code].shape[1]//2):
            for traj_r in range(data[code].shape[1]//2,
                                data[code].shape[1]):

                pr = pearsonr(data[code][grid,traj_l,:,0],
                              data[code][grid,traj_r,:,0]
                               )[0]
                corr_matrix.append(pr)
        corr_matrices.append(corr_matrix)
    corr_dict[code] = [np.array(x).mean() for x in corr_matrices]

df = pd.DataFrame.from_dict(corr_dict)
df['grid_seed'] = df.index
df = df.melt(id_vars=["grid_seed"], var_name = "code_sim", value_name = "pearsonr")

codings = df['code_sim'].str.split(pat='_', expand=True)
codings.columns = ["code", "whatever", "similarity"]
#codings['grid_seed'] = df['grid_seed']
df = df.join(codings, how='inner')
plt.plot()
plt.close('all')
sns.violinplot(x='code', y='pearsonr', hue="similarity", data =df)


# Calculate variance
codes = ['rate_code_diff', 'phase_code_diff', 'complex_code_diff',
         'rate_code_sim', 'phase_code_sim', 'complex_code_sim']
diff_dict = {}
for code in codes:
    diff_matrices = []
    for grid in range(data[code].shape[0]):
        diff_matrix = []
        for traj_l in range(data[code].shape[1]//2):
            for traj_r in range(data[code].shape[1]//2,
                                data[code].shape[1]):

                pr = np.abs(data[code][grid,traj_l,:,0] -
                            data[code][grid,traj_r,:,0])

                diff_matrix.append(pr)
        diff_matrices.append(diff_matrix)
    diff_dict[code] = [np.array(x).mean() for x in diff_matrices]

df_diff = pd.DataFrame.from_dict(diff_dict)
df_diff['grid_seed'] = df_diff.index
df_diff = df_diff.melt(id_vars=["grid_seed"], var_name = "code_sim", value_name = "difference")
df['difference']  = df_diff['difference']

plt.figure()
sns.violinplot(x='code', y='difference', hue="similarity", data =df)








