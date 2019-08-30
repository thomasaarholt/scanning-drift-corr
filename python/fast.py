import numpy as np
from numpy.fft import fft2
import matplotlib.pyplot as plt

from pathlib import Path
import h5py
from scipy.ndimage import fourier_shift

from transforms import (
    hybrid_correlation,
    phase_correlation,
    cross_correlation,
    phase_translation,
    cross_translation,
    pad_images,
    set_shear_and_scale_ranges,
    set_transform_matrices,
    transform,
    transform_single_image,
    normalise_max,
)

pad_scale = 1.1
steps = 5
correlation_method = "cross"
shear_steps = steps
scale_steps = steps

if correlation_method == "phase":
    correlation_function = phase_correlation
    translation_function = phase_translation
elif correlation_method == "hybrid":
    correlation_function = phase_correlation
    translation_function = phase_translation
elif correlation_method == "cross":
    correlation_function = cross_correlation
    translation_function = cross_translation
else:
    raise ValueError("No correlation method with that name")

test_dataset = "C"

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
    images = [s.data for s in signals]
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
    img2 = np.fft.ifftn(fourier_shift(fft2(img1), SHIFT)).real
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
    img3 = f["imageIdeal"][:]
    images = [img1, img2]
    angles = [0, 90]

else:
    print("Specified wrong data?")

images = [normalise_max(img) for img in images]
padded_images, weights = pad_images(images, pad_scale=pad_scale)

# Set the various sheares, scales, first in terms of range
sheares, scales = set_shear_and_scale_ranges(
    padded_images[0].shape, shear_steps=shear_steps, scale_steps=scale_steps, pix=1
)
# Then in terms of transform matrices
transform_matrices = set_transform_matrices(angles, sheares, scales)
rotation_matrices, shear_matrices, scale_matrices = transform_matrices

# Scale and shear the masked data
data = transform(
    padded_images, rotation_matrices, shear_matrices, scale_matrices, weights=weights
)
data = data.reshape((len(padded_images), -1) + padded_images[0].shape)
data = data.swapaxes(0, 1)

correlations = []
for i, image_sequence in enumerate(data):
    img1 = image_sequence[0]
    corr = [
        correlation_function(fft2(weights * img1), fft2(weights * img2))
        for img2 in image_sequence[1:]
    ]
    correlations.append(corr)

correlations2 = np.array(correlations).swapaxes(0, 1)

max_indexes = []
shifts_list = []
for corr2 in correlations2:
    max_indexes.append(np.argmax([np.max(corr) for corr in corr2]))
    shifts_list.append([translation_function(corr) for corr in corr2])

# Plot masked images
data2 = np.array(data).swapaxes(0, 1)
image_sums = []
for i, img_array in enumerate(data2[1:]):
    max_index = max_indexes[i]
    img1 = data2[0, max_index]
    img2 = img_array[max_index]
    img2_shifted = np.fft.ifftn(
        fourier_shift(fft2(img2), shifts_list[i][max_index])
    ).real
    image_sums.append(img1 + img2_shifted)

fig, AX = plt.subplots(ncols=len(padded_images) - 1, squeeze=False)
for i, ax in enumerate(np.reshape(AX, np.prod(AX.shape))):
    ax.imshow(image_sums[i], cmap="viridis")
    ax.axis("off")

# Plot whole images
fig, AX = plt.subplots(ncols=len(padded_images) - 1, squeeze=False)

# Loop across 0, numImages - 1
for i, ax in enumerate(np.reshape(AX, np.prod(AX.shape))):
    shear_index, scale_index = np.unravel_index(
        max_indexes[i], (shear_steps, scale_steps)
    )
    shear, scale = sheares[shear_index], scales[scale_index]

    angle = [angles[0]]  # first image and i+1 image
    rot_matrices, shear_matrices, scale_matrices = set_transform_matrices(
        [angle], [shear], [scales]
    )
    img1 = transform_single_image(
        padded_images[0], rot_matrices[0], shear_matrices[0], scale_matrices[0]
    )

    angle = [angles[i + 1]]  # first image and i+1 image

    rot_matrices, shear_matrices, scale_matrices = set_transform_matrices(
        [angle], [shear], [scales]
    )

    img2 = transform_single_image(
        padded_images[i + 1], rot_matrices[0], shear_matrices[0], scale_matrices[0]
    )

    ax.imshow(img1 + img2, cmap="viridis")
    ax.axis("off")
plt.show()
