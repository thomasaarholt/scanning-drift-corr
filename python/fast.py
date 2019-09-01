import numpy as np
from numpy.fft import fft2
import matplotlib.pyplot as plt
import time
from pathlib import Path
import h5py
from scipy.ndimage import fourier_shift
import logging

start_time = time.time()

from transforms import (
    prepare_correlation_data,
    correlate_images,
    correlation,
    translate,
    pad_images,
    set_shear_and_scale_ranges,
    set_transform_matrices,
    transform,
    transform_single_image,
    plot_transformed_images,
    normalise_max,
)

pad_factor = 1.3  # larger than 1, approx 1.25 is good
steps = 11  # odd, 5,7 is normal
correlation_method = "phase"  # "phase", "cross", "hybrid"
gpu = True  # True / False
shear_steps = steps
scale_steps = steps

test_dataset = "A"

if gpu:
    import tensorflow as tf

    tf.device("/gpu:0")

if test_dataset == "A":
    import hyperspy.api as hs

    def get_haadf(slist):
        "Helper function for picking out HAADF from velox format"
        for s in slist:
            if "HAADF" in s.metadata.General.title:
                return s

    folder = r"C:\Users\Me\Documents\STEM Images"
    signals = [
        get_haadf(hs.load(str(f))) for f in Path(folder).iterdir() if f.is_file()
    ]
    images = [s.data.astype("float32") for s in signals]
    angles = [
        float(s.original_metadata.Scan.ScanRotation) * 180 / np.pi for s in signals
    ]


elif test_dataset == "B":
    from scipy.misc import face, ascent
    from scipy.ndimage import rotate

    img1 = face(True)  # [:,:768]
    img1 = np.pad(img1, 200, "constant")
    SHIFT = [80, 90]
    ANGLE = 45
    img2 = np.fft.ifft2(fourier_shift(fft2(img1), SHIFT)).real
    img2 = rotate(img2, ANGLE, reshape=False)
    images = [img1, img2]
    angles = (0, ANGLE)

elif test_dataset == "C":
    import h5py

    file = (
        "../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat"
    )
    f = h5py.File(file, mode="r")
    img1 = f["image00deg"][:]
    img2 = f["image90deg"][:]
    # img3 = f["imageIdeal"][:]
    images = [img1, img2]
    angles = [0, 90]

else:
    print("Specified wrong data?")

images = [normalise_max(img) for img in images]
print("Padding images")
padded_images, weights = pad_images(images, pad_factor=pad_factor)
GB = np.round(
    float(np.prod(np.shape(padded_images)))
    * shear_steps
    * scale_steps
    * padded_images[0].dtype.itemsize
    / 1e9,
    2,
)  # 32 bytes per complex64

print("Estimating memory usage of {}GB".format(GB))
# Set the various sheares, scales, first in terms of range
sheares, scales = set_shear_and_scale_ranges(
    padded_images[0].shape, shear_steps=shear_steps, scale_steps=scale_steps, pix=2
)
# Then in terms of transform matrices
print("Calculating transform matrices")
transform_matrices = set_transform_matrices(angles, sheares, scales)

rotation_matrices, shear_matrices, scale_matrices = transform_matrices

# Scale and shear the masked data
print("Transforming data")
data = transform(
    padded_images, rotation_matrices, shear_matrices, scale_matrices, weights=weights
)

data = data.astype("float32")

data = data.reshape((len(padded_images), -1) + padded_images[0].shape)
data = data.swapaxes(0, 1)

print("Preparing correlation data")
correlation_data = prepare_correlation_data(
    data, weights, method=correlation_method, gpu=gpu
)

print("Correlating")
max_indexes, shifts_list = correlate_images(
    correlation_data, method=correlation_method, gpu=gpu
)
# Plot masked images
print("Calculating final images")
data2 = np.array(data).swapaxes(0, 1)
image_sums = []

for i, img_array in enumerate(data2[1:]):
    max_index = int(max_indexes[i])
    img1 = data2[0, max_index]
    img2 = img_array[max_index]
    shift = shifts_list[i][max_index]
    img2_shifted = np.fft.ifftn(fourier_shift(fft2(img2), shift)).real
    image_sums.append(img1 + img2_shifted)
    print("Image drifted ({}, {}) pixels since first frame".format(shift[0], shift[1]))
# fig, AX = plt.subplots(ncols=len(padded_images) - 1, squeeze=False)
# for i, ax in enumerate(np.reshape(AX, np.prod(AX.shape))):
#     ax.imshow(image_sums[i], cmap="viridis")
#     ax.axis("off")

plot_transformed_images(
    padded_images,
    angles,
    shifts_list,
    max_indexes,
    sheares,
    scales,
    shear_steps,
    scale_steps,
)

print("--- %s seconds ---" % (time.time() - start_time))
