import numpy as np
import matplotlib.pyplot as plt
import h5py

from scan import SPmerge01

file = "../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat"
f = h5py.File(file, mode='r')
img1 = f['image00deg'][:]
img2 = f['image90deg'][:]
img3 = f['imageIdeal'][:]

imgs = [img1, img2, img1]

#imgs = [img[:, :-5] for img in imgs]
scanAngles = [0, 90, 0]

# fig, AX = plt.subplots(ncols=3, figsize=(15, 5))
# #fig.canvas.layout.width = '500px'
# #fig.canvas.layout.height = '300px'

# for ax, img, in zip(AX.flatten(), imgs):
#     ax.imshow(img)
#     ax.axis('off')
# plt.show()

SPmerge01(imgs, scanAngles)

print('hi')
