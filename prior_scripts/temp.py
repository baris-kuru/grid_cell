import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

plt.close('all')

"""
for odd freqs it causes reverse intersection of negative peaks of gratings 
and therefore negative bumps in the grid

calculations like np.cos(w3*t*interval3)*2 or 
np.cos(w2*t*interval2-2) changes the amplitude

changing interval effects the phase but bumps only occur 
in certain conditions and doesnt occur in when we shift the pase
since we use 3 individual seperate gratings 

to change the phase of all we should shift the elements of resulting grid
np.roll solved this

for some freqs rolling makes it look weird bcs it shifts the end and reintroduce 
it to the first

rolling for axis=1 always works well tho? 

check also np.rollaxis and .shape, might help """

dims = np.array([1000,1000])
arr = np.ones(dims)
arr2 = np.ones(dims)
arr3 = np.ones(dims)
arr4 = np.ones(dims)
arr5 = np.ones(dims)
f = 24
f2 = 26
f3 = 28
f4 = 30
f5 = 34
w = 2*np.pi*f # I removed the 2* here
w2 = 2*np.pi*f2
w3 = 2*np.pi*f3
w4 = 2*np.pi*f4
w5 = 2*np.pi*f5
t = 1 # t has no effect
interval = np.linspace(0,1,dims[1])
interval2 = np.linspace(0,1,dims[1])
interval3 = np.linspace(0,1,dims[1])
interval4 = np.linspace(0,1,dims[1])
interval5 = np.linspace(0,1,dims[1])

sin_wave = np.cos(w*t*interval)
sin_wave2 = np.cos(w2*t*interval2)
sin_wave3 = np.cos(w3*t*interval3)
sin_wave4 = np.cos(w4*t*interval4)
sin_wave5 = np.cos(w5*t*interval5)


arr = arr*sin_wave
arr_60 = ndimage.rotate(arr, 60, reshape=False, mode='wrap')
arr_120 = ndimage.rotate(arr, 120, reshape=False, mode='wrap')
grid_arr = 2/3*((arr + arr_60 + arr_120)/3 + 1/2) #calc here has no effect
grid_arr = (arr + arr_60 + arr_120)
#grid_arr = np.roll(grid_arr, 100, axis=1)

arr2 = arr2*sin_wave2
arr2_60 = ndimage.rotate(arr2, 60, reshape=False, mode='wrap')
arr2_120 = ndimage.rotate(arr2, 120, reshape=False, mode='wrap')
grid_arr2 = 2/3*((arr2 + arr2_60 + arr2_120)/3 + 1/2)
grid_arr2 = (arr2 + arr2_60 + arr2_120)
#grid_arr2 = np.roll(grid_arr2, 100, axis=0) #changing phase of the grid

arr3 = arr3*sin_wave3
arr3_60 = ndimage.rotate(arr3, 60, reshape=False, mode='wrap')
arr3_120 = ndimage.rotate(arr3, 120, reshape=False, mode='wrap')
grid_arr3 = 2/3*((arr3 + arr3_60 + arr3_120)/3 + 1/2)
grid_arr3 = (arr3 + arr3_60 + arr3_120)
#grid_arr3 = np.roll(grid_arr3, 70, axis=1)


arr4 = arr4*sin_wave4
arr4_60 = ndimage.rotate(arr4, 60, reshape=False, mode='wrap')
arr4_120 = ndimage.rotate(arr4, 120, reshape=False, mode='wrap')
grid_arr4 = 2/3*((arr4 + arr4_60 + arr4_120)/3 + 1/2)
grid_arr4 = (arr4 + arr4_60 + arr4_120)
#grid_arr4 = np.roll(grid_arr4, 50, axis=1)

arr5 = arr5*sin_wave5
arr5_60 = ndimage.rotate(arr5, 60, reshape=False, mode='wrap')
arr5_120 = ndimage.rotate(arr5, 120, reshape=False, mode='wrap')
grid_arr5 = 2/3*((arr5 + arr5_60 + arr5_120)/3 + 1/2)
grid_arr5 = (arr5 + arr5_60 + arr5_120)
#grid_arr5 = np.roll(grid_arr5, 900, axis=1)

sum_grid = grid_arr + grid_arr2 + grid_arr3 + grid_arr4 + grid_arr5

#kernel5 = np.array([[-2, -2, -2, -2, -2],
#                   [-2,  2,  4,  2, -2],
#                   [-1,  4,  15,  4, -2],
#                   [-2,  2,  4,  2, -2],
#                   [-2, -2, -2, -2, -2]])
#
#kernel3 = 10*np.array([[0, -0.5, 0],
#                   [-0.5, 3, -0.5],
#                   [0,-0.5, 0]] )
#
#sharp = ndimage.convolve(grid_arr, kernel3)

#low_pass = ndimage.gaussian_filter(grid_arr, 25)
#
#high_pass = grid_arr - high_pass_5
#high_pass = grid_arr - low_pass
#plt.imshow(arr, cmap='jet')
#plt.figure()
#plt.imshow(arr_60, cmap='jet')
#plt.figure()
#plt.imshow(arr_120, cmap='jet')
#plt.figure()
#plt.imshow(grid_arr, cmap='plasma')
#plt.figure()
#plt.imshow(high_pass, cmap='plasma')
plt.imshow(grid_arr, cmap='jet')
plt.figure()
plt.imshow(grid_arr2, cmap='jet')
plt.figure()
plt.imshow(grid_arr3, cmap='jet')
plt.figure()
plt.imshow(grid_arr4, cmap='jet')
plt.figure()
plt.imshow(grid_arr5, cmap='jet')
plt.figure()
plt.imshow(sum_grid, cmap='jet')
#plt.figure()
#fig, axes = plt.subplots(1, 3)
#axes[0, 0].imshow(arr)
#axes[0, 1].imshow(arr_60)
#axes[0, 2].imshow(arr_120)