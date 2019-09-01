import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

import logging
import os
from tqdm.auto import tqdm

from skimage.transform import AffineTransform, warp
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve

try:
    import tensorflow as tf

    from tensorflow_transforms import (
        cross_correlation_tf,
        phase_correlation_tf,
        hybrid_correlation_tf,
    )
except:
    logging.warning("Tensorflow not available, do not use gpu=True")


def prepare_correlation_data(data, weights, method="phase", gpu=True):
    if method != "cross":
        if gpu:
            data = data.astype("complex64")
            fftdata = tf.signal.fft2d(data)
            return fftdata
        else:
            return fft2(data)
    else:
        if gpu:
            return tf.convert_to_tensor(data)
        else:
            return data


def correlate_images(correlation_data, method="phase", gpu=True):
    correlations = []
    for image_sequence in tqdm(correlation_data):
        img1 = image_sequence[0]
        corr = [
            correlation(img1, img2, method=method, gpu=gpu)
            for img2 in image_sequence[1:]
        ]

        correlations.append(corr)
    if gpu:
        correlations = tf.transpose(correlations, [1, 0, 2, 3])
    else:
        correlations = np.array(correlations).swapaxes(0, 1)

    max_indexes = []
    shifts_list = []

    if gpu:
        argmax = tf.math.argmax
        max_ = tf.math.reduce_max
    else:
        argmax = np.argmax
        max_ = np.max
    for transformed_images in correlations:
        value_of_pixels_with_highest_correlation = [
            max_(corr) for corr in transformed_images
        ]
        transform_with_best_match = np.array(
            argmax(value_of_pixels_with_highest_correlation)
        )
        max_indexes.append(transform_with_best_match)
        shifts_list.append(
            [translate(corr, method=method, gpu=gpu) for corr in transformed_images]
        )
    return max_indexes, shifts_list


def correlation(img1, img2, method="phase", gpu=True):
    if method == "phase":
        if gpu:
            correlation_function = phase_correlation_tf
        else:
            correlation_function = phase_correlation

    elif method == "hybrid":
        if gpu:
            correlation_function = phase_correlation_tf
        else:
            correlation_function = phase_correlation

    elif method == "cross":
        if gpu:
            correlation_function = cross_correlation_tf
        else:
            correlation_function = cross_correlation
    else:
        raise ValueError("No correlation method with that name")

    return correlation_function(img1, img2)


def translate(corr, method="phase", gpu=True):
    if method == "cross":
        translation_function = cross_translation
    else:
        translation_function = phase_translation
    return translation_function(corr, gpu=gpu)


def hybrid_correlation(img1_fft, img2_fft):
    """Unsure if this actually the hybrid correlation Ophus refers to.
    Works on images already in fourier space.
    """
    m = img1_fft * np.conj(img2_fft)

    magnitude = np.sqrt(np.abs(m))
    euler = np.exp(1j * np.angle(m))
    Icorr = np.fft.ifft2(magnitude * euler).real
    return Icorr


def phase_correlation(img1_fft, img2_fft):
    "Perform phase correlation on images that are already in fourier space"
    C = img1_fft * np.conj(img2_fft)
    return np.fft.ifft2(C / np.abs(C)).real


def cross_correlation(im1, im2):
    "Perform cross correlation on images in real space"
    # get rid of the averages, otherwise the results are not good
    im1 -= np.mean(im1)
    im2 -= np.mean(im2)

    # calculate the correlation image; note the flipping of the images
    return fftconvolve(im1, im2[::-1, ::-1], mode="same")


def changespace(x, maxval):
    """Shift indices from [0, max] to [-max/2, max/2]
    Handles both single values and numpy shape tuples
    """
    x = np.array(x)
    maxval = np.array(maxval)
    half = maxval / 2
    return (x + half) % maxval - half


def phase_translation(corr, gpu=True):
    "Turn phase correlation array into shift x,y"
    if gpu:
        argmax = tf.math.argmax
        unravel_index = tf.unravel_index

        def flatten(x):
            return tf.reshape(x, [-1])

    else:

        def flatten(x):
            return x.flatten()

        argmax = np.argmax
        unravel_index = np.unravel_index

    shift = unravel_index(argmax(flatten(corr)), corr.shape)
    shift = changespace(shift, corr.shape)
    return shift


def cross_translation(corr, gpu=True):
    "Turn phase correlation array into shift x,y"
    if gpu:
        argmax = tf.math.argmax
        unravel_index = tf.unravel_index

        def flatten(x):
            return tf.reshape(x, [-1])

    else:
        argmax = np.argmax
        unravel_index = np.unravel_index
        flatten = np.flatten

    shift = unravel_index(argmax(flatten(corr)), corr.shape)
    return shift - np.array(corr.shape) / 2


def pad_images(images, pad_factor=1.25):
    """Pad images as a function of their size.
    Also calculates a window function, also padded.
    """
    old_shape = np.array(images[0].shape)
    new_shape = old_shape * pad_factor  # approx
    padding = ((new_shape - old_shape) / 2).astype(int)
    padding = [(padding[0], padding[0]), (padding[1], padding[1])]
    padded_images = [
        np.pad(img, padding, "constant", constant_values=0) for img in images
    ]
    new_shape = np.array(padded_images[0].shape)

    weights = HanningImage(new_shape)

    # weights = HanningImage(old_shape[0]) * HanningImage(old_shape[1]).T
    # weights = np.pad(weights, padding, mode="constant", constant_values=0)
    return padded_images, weights


def set_shear_and_scale_ranges(image_shape, shear_steps=5, scale_steps=5, pix=1):
    "Turn a number of shear and scale steps into a range of - to +."
    # Scale so `pix` pixels wider with each step
    scale = (image_shape[1] + pix) / image_shape[1] - 1
    # Shear by an angle equivalent of shearing `pix` pixels

    half_steps = (scale_steps - 1) / 2
    scale_limits = scale * half_steps
    scales = 1 + np.linspace(-scale_limits, scale_limits, scale_steps)

    half_steps = (shear_steps - 1) / 2
    shear = np.arctan(pix / image_shape[1])  # maybe scale[0]?
    shear_limits = shear * half_steps
    sheares = np.linspace(-shear_limits, shear_limits, shear_steps)
    return sheares, scales


def set_transform_matrices(angles, sheares, scales):
    """Take a list of angles, shearing values and scaling values,
    and convert them to equivalent affine transformation matrices
    """
    rotation_matrices = [
        AffineTransform(rotation=np.deg2rad(angle)) for angle in angles
    ]
    shear_matrices = [AffineTransform(shear=shear) for shear in sheares]
    scale_matrices = [AffineTransform(scale=(1, scale)) for scale in scales]
    return rotation_matrices, shear_matrices, scale_matrices


def transform(
    padded_images, rotation_matrices, shear_matrices, scale_matrices, weights=1, steps=5
):
    """Scales, then shears, and then rotates images according the
    angles. If no window is given as a weights argument, the function
    operates directly on the raw images.
    
    """
    shape = padded_images[0].shape
    # Get shifts so that we can shift the image by half the image
    # size and then rotate about the centre instead of the corner
    # (and then shift back afterwards)
    shift_x, shift_y = np.array(shape) / 2.0
    tf_shift = AffineTransform(translation=[-shift_y, -shift_x])
    tf_shift_inv = AffineTransform(translation=[shift_y, shift_x])

    data = np.zeros(
        (len(padded_images), len(shear_matrices), len(scale_matrices)) + shape
    )
    for i, (img, tf_rotate) in enumerate(zip(padded_images, rotation_matrices)):
        for j, tf_shear in enumerate(shear_matrices):
            for k, tf_scale in enumerate(scale_matrices):
                data[i, j, k] = weights * warp(
                    img,
                    (tf_shift + tf_scale + tf_shear + tf_rotate + tf_shift_inv).inverse,
                    order=0,
                )

    return data


def transform_single_image(img, rot_matrix, shear_matrix, scale_matrix, weights=1):
    """Equivalent of transform, but for a single transformation matrix rather than a list
    """
    shift_x, shift_y = np.array(img.shape) / 2.0
    tf_shift = AffineTransform(translation=[-shift_y, -shift_x])
    tf_shift_inv = AffineTransform(translation=[shift_y, shift_x])
    return weights * warp(
        img,
        (tf_shift + scale_matrix + shear_matrix + rot_matrix + tf_shift_inv).inverse,
        order=0,
    )


def plot_transformed_images(
    padded_images,
    angles,
    shifts_list,
    max_indexes,
    sheares,
    scales,
    shear_steps,
    scale_steps,
):
    "Apply the best transformation to each image, and plot them on top of each other"

    # Plot whole images

    fig, AX = plt.subplots(ncols=len(padded_images) - 1, squeeze=False)

    for i, ax in enumerate(AX.flatten()):
        max_index = max_indexes[i]
        shear_index, scale_index = np.unravel_index(
            max_index, (shear_steps, scale_steps)
        )
        shear, scale = sheares[shear_index], scales[scale_index]
        shift = shifts_list[i][max_index]
        rot_mat, shear_mat, scale_mat = set_transform_matrices(
            [angles[0]], [shear], [scale]
        )

        img1 = transform_single_image(
            padded_images[0], rot_mat[0], shear_mat[0], scale_mat[0]
        )

        rot_mat, shear_mat, scale_mat = set_transform_matrices(
            [angles[i + 1]], [shear], [scale]
        )
        img2 = transform_single_image(
            padded_images[i + 1], rot_mat[0], shear_mat[0], scale_mat[0]
        )
        img2_shift = ifft2(fourier_shift(fft2(img2), shift)).real

        ax.imshow(img1 + img2_shift, cmap="viridis")
        ax.axis("off")
    return img1, img2_shift
    # plt.show()


def HanningImage(shape):
    "Return a 2D hanning window of size (x, y)"
    x, y = shape
    hanx = np.hanning(x)
    hany = np.hanning(y)
    return (hanx * hany[:, None]).T


def normalise_max(img):
    "Use to normalise before correlation. Unsure if necessary or a good idea"
    return img / img.max()
