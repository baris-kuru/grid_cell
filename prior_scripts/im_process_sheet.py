#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:19:13 2020

@author: bariskuru
"""


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")

image = image.convert('1')

data = np.asarray(image)

image2 = Image.fromarray(data2)

data = np.asarray(image2)

data2 =signal.resample(data, 4096, axis=1)
data2 =signal.resample(data2, 3072, axis=0)

plt.close('all')

plt.imshow(data2)
