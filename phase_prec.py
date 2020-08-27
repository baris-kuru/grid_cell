
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_trajs import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line


spacing = 50
arr = grid_maker(spacing, 0, [100,100], 200, [1,1], 20, seed_1=1)
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

def sine(val, time):
    phase = np.pi*val
    amp = (np.sin(8*(2*np.pi-phase)*time)+1)/2
    return amp



plt.close('all')
traj_rate = profile_line(arr, (99,0), (99,200-1))
trans_norm_2d = np.arccos(3*arr/40-0.5)/2
traj_phase = profile_line(trans_norm_2d, (99,0), (99,200-1))
dt = 0.025


f=8
T = 1/f
dt = 0.025
time = np.arange(0,5,dt)
time_hr = np.arange(0,5,0.0001)
shift = 3*np.pi/2
theta = (np.sin(f*2*np.pi*time+shift)+1)/2
theta_hr = (np.sin(f*2*np.pi*time_hr+shift)+1)/2

phase = (2*np.pi*(time%T)/T + shift)%(2*np.pi)


direction = np.diff(traj_phase)

#last element is same with the -1 element of diff array
direction = np.append(direction, direction[-1])
plt.plot(direction)


plt.figure()
plt.plot(traj_phase)
direction[direction < 0] = -1
direction[direction > 0] = 1
direction = -direction

# direction defines if animal goes into a grid field or goes out of a grid field
traj_phase_dir = traj_phase*direction
plt.figure()
plt.plot(traj_phase_dir)

firing_phase = 2*np.pi*(traj_phase+0.5)

firing_phase_dir = 2*np.pi*(traj_phase_dir+0.5)

plt.figure()
plt.plot(phase)
plt.figure()
plt.plot(firing_phase)
plt.figure()
plt.plot(firing_phase_dir)


phase_code = np.exp(1.5*np.cos(firing_phase-phase))

phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase))

plt.figure()

time_plt = np.arange(0, 5, 0.025)
plt.plot(phase_code)
plt.figure()
plt.plot(time_plt, phase_code_dir)

overall = phase_code*traj_rate
plt.figure()
plt.plot(overall)

overall_dir = phase_code_dir*traj_rate
plt.figure()
plt.plot(overall_dir)

plt.figure()
plt.plot(phase)
plt.plot(theta)
plt.plot(time_hr, 4*theta_hr)

plt.figure()
plt.plot(traj_rate)




###### poisson spikes
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
dt_s = 0.0001
t_sec = 5
asig = AnalogSignal(overall_dir,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=t_sec*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            #generate the spike train out of analog signal
curr_train = stg.inhomogeneous_poisson_process(asig)
curr_train2 = stg.inhomogeneous_poisson_process(asig)
curr_train3 = stg.inhomogeneous_poisson_process(asig)
curr_train4 = stg.inhomogeneous_poisson_process(asig)
curr_train5 = stg.inhomogeneous_poisson_process(asig)

curr_train = []
for i in range(10):
    curr_train.append(stg.inhomogeneous_poisson_process(asig))
    
plt.eventplot(curr_train, lineoffsets=[1,2,3,4,5,6,7,8,9,10])
plt.eventplot(curr_train2)
plt.eventplot(curr_train3)
plt.eventplot(curr_train4)
plt.eventplot(curr_train5)



