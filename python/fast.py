import numpy as np
from numpy.fft import fft2

import matplotlib.pyplot as plt
import h5py
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter, fourier_shift
from ipywidgets import widgets as w
from IPython.display import display

from transforms import (
    cross_correlation,
    translation,
    translation_orig,
    pad_images,
    transform,
    normalise_max,
    setsize,
    is_notebook,
)

file = "../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat"
f = h5py.File(file, mode="r")
img1 = f["image00deg"][:]
img2 = f["image90deg"][:]
img3 = f["imageIdeal"][:]
images = [img1, img2]
angles = [0, 90]

from scipy.misc import face, ascent
from scipy.ndimage import rotate, gaussian_filter

if True:
    img1 = face(True)  # [:,:768]
    img1 = ascent()
    img1 = np.pad(img1, 200, "constant")
    SHIFT = [80, 90]
    ANGLE = 45
    img2 = np.fft.ifftn(fourier_shift(fft2(img1), SHIFT)).real
    img2 = rotate(img2, ANGLE, reshape=False)
    images = [img1, img2]
    angles = (0, ANGLE)

images = [normalise_max(img) for img in images]
padded_images, weights = pad_images(images)

data_raw = transform(padded_images, angles, weights=1)
data_raw[1] = data_raw[1].swapaxes(0, 1)
data_raw = data_raw.reshape((2, -1) + padded_images[0].shape)
data_raw = data_raw.swapaxes(0, 1)

data = transform(padded_images, angles, weights=weights)
data[1] = data[1].swapaxes(0, 1)
data = data.reshape((2, -1) + padded_images[0].shape)
# s1, s2 = [hs.signals.Signal2D(d) for d in data]
data = data.swapaxes(0, 1)

centre_index = (len(data) - 1) // 2

o1, o2 = w.Output(), w.Output()
if is_notebook():
    with o1:
        fig, ax1 = plt.subplots()
        setsize(fig, 500)
    with o2:
        fig, ax2 = plt.subplots()
        setsize(fig, 500)
else:
    fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.imshow(data_raw[centre_index, 0], cmap="viridis")
ax2.imshow(data_raw[centre_index, 1], cmap="viridis")
h = w.HBox([o1, o2])
display(h)


correlations = [
    cross_correlation(fft2(weights * img1), fft2(weights * img2)) for img1, img2 in data
]

max_index = np.argmax([np.max(corr) for corr in correlations])
shifts = [translation_orig(corr) for corr in correlations]
shifts[max_index]

o1, o2 = w.Output(), w.Output()
if is_notebook():
    with o1:
        fig, ax1 = plt.subplots()
        setsize(fig)
    with o2:
        fig, ax2 = plt.subplots()
        setsize(fig)
else:
    fig, (ax1, ax2) = plt.subplots(ncols=2)

img1, img2 = data[max_index]
img2_shifted = np.fft.ifftn(fourier_shift(fft2(img2), shifts[max_index])).real
ax1.imshow(img1 + img2_shifted, cmap="viridis")

img1, img2 = data_raw[max_index]
img2_shifted = np.fft.ifftn(fourier_shift(fft2(img2), shifts[max_index])).real
ax2.imshow(img1 + img2_shifted, cmap="viridis")
print(shifts[max_index])
plt.show()
h = w.HBox([o1, o2])
display(h)
