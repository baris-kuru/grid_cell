image.shape
image.format
data = asarray(image)
import numpy as np
data = asarray(image)
data = np.asarray(image)
data.type
data.format
data.shape
plt.imshow(data)
data2 =signal.resample(data, 4096, axis=1)
plt.imshow(data2)
data2 =signal.resample(data, 2400, axis=1)
plt.imshow(data2)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)


data2 =signal.resample(data, 2048, axis=1)

plt.close('all')
plt.imshow(data2)

image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)


data2 =signal.resample(data, 1000, axis=1)

plt.close('all')
plt.imshow(data2)
min(data2)
data2
data
image2 = Image.fromarray(data)
image2
image2.format
image2.mode
data = np.asarray(image2)
plt.imshow(data)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)

image2 = Image.fromarray(data)

data = np.asarray(image2)
data.shape
plt.imshow(data)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)

image2 = Image.fromarray(data)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)
plt.imshow(data)
data.shape
type(data)
image = image.convert('1')
data = np.asarray(image)
data.shape
data2 =signal.resample(data, 2048, axis=1)
plt.imshow(data2)
data2 =signal.resample(data, 2500, axis=1)
plt.imshow(data2)
data2 =signal.resample(data, 2048, axis=1)
data2 =signal.resample(data2, 1536, axis=0)
plt.close('all')

plt.imshow(data2)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")

image = image.convert('1')

data = np.asarray(image)

plt.imshow(data)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")
data = np.asarray(image)
data.shape
data2 =signal.resample(data, 2048, axis=1)
data2 =signal.resample(data2, 1536, axis=0)
plt.imshow(data)
plt.imshow(data2)
data2 =int(signal.resample(data, 2048, axis=1))
data2 =int(signal.resample(data2, 1536, axis=0))
image2 = Image.fromarray(data2)
image = Image.open("/Users/bariskuru/Desktop/OldPhotos/19092010059.jpg")

image = image.convert('1')
data2 =signal.resample(data, 4096, axis=1)
data2 =signal.resample(data2, 3072, axis=0)

plt.close('all')

plt.imshow(data2)