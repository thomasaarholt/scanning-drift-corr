import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage.transform import rotate
from scipy.ndimage import shift

from scan import SPmerge01

file = "../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat"
f = h5py.File(file, mode="r")
img1 = f["image00deg"][:]
img2 = f["image90deg"][:]
img3 = f["imageIdeal"][:]
scanAngles = [0, 90, 0]
imgs = [img1, img2, img1]

if False:
    img1b = rotate(img1, 30, True)
    img1c = shift(img1b, [5, 30])

    pad = (img1b.shape[0] - img1.shape[0]) / 2
    img1a = np.pad(img1, int(pad), "constant", constant_values=0)

    img1 = img1a
    img2 = img1c
    scanAngles = [0, -30]
    imgs = [img1, img2]

imgs = [img.T for img in imgs]
# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.imshow(img1)
# ax2.imshow(img2)
# fig.suptitle("orig")
# plt.show()

# imgs = [img[:, :-5] for img in imgs]


# fig, AX = plt.subplots(ncols=3, figsize=(15, 5))
# #fig.canvas.layout.width = '500px'
# #fig.canvas.layout.height = '300px'

# for ax, img, in zip(AX.flatten(), imgs):
#     ax.imshow(img)
#     ax.axis('off')
# plt.show()

SPmerge01(imgs, scanAngles)
