import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.transform import rotate
from scipy.ndimage import shift

from scipy.misc import face

from alignment import rough

file = "../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat"
f = h5py.File(file, mode="r")
img1 = f["image00deg"][:]
img2 = f["image90deg"][:]
img3 = f["imageIdeal"][:]
imgs = [img1, img2]
angles = [0, 90]

# img1 = np.pad(face(True), 50, "constant", constant_values=0)
# img2 = np.roll(img1, 20, axis=0)
# imgs = [img1, img2]

rough(imgs, angles)
