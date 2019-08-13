import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve
from time import time


def cross_correlation(img1_fft, img2_fft):
    m = img1_fft * np.conj(img2_fft)

    magnitude = np.sqrt(np.abs(m))
    euler = np.exp(1j * np.angle(m))
    Icorr = np.fft.ifft2(magnitude * euler).real
    return Icorr

def translation(corr):
    translation = np.unravel_index(corr.argmax(), corr.shape)
    halfshape = np.array(corr.shape) / 2
    translation = (translation[::-1] - halfshape) % halfshape
    return translation

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def hanningLocal(N):
    # Should this be 1, N+1? Or 0, N?
    Dim1Window = np.sin(np.pi * np.arange(1, N + 1) / (N + 1)) ** 2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def rough(images, angles):

    pad_scale = 1.25
    old_shape = np.array(images[0].shape)
    new_shape = old_shape * pad_scale  # approx
    padding = ((new_shape - old_shape) / 2).astype(int)
    padded_images = [
        np.pad(img, padding, "constant", constant_values=0) for img in images
    ]

    new_shape = np.array(padded_images[0].shape)

    rotated_images = [rotate(img, -angle) for img, angle in zip(padded_images, angles)]

    TwoDHanningWindow = hanningLocal(old_shape[0]) * hanningLocal(old_shape[1]).T

    weights = np.pad(TwoDHanningWindow, padding, mode="constant", constant_values=0)

    weighted_images = weights * rotated_images

    first_imgs, last_imgs = weighted_images[:-1], weighted_images[1:]

    first_ffts = [np.fft.fft2(weights * img) for img in first_imgs]
    last_ffts = [np.fft.fft2(weights * img) for img in last_imgs]

    correlations = [
        cross_correlation(fft1, fft2) for fft1, fft2 in zip(first_ffts, last_ffts)
    ]

    translations = [
        translation(corr) for corr in correlations
    ]
    print(translations)
    corrected_images = [rotated_images[0]] + [
        np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)).real
        for img, shift in zip(rotated_images[1:], translations)
    ]

    fig, AX = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    imgs = corrected_images + [
        np.mean(rotated_images, axis=0),
        np.mean(corrected_images, axis=0),
    ]
    for ax, im in zip(AX.flatten(), imgs):
        ax.imshow(im)
    plt.show()

    # plt.figure()
    # plt.imshow(np.mean(corrected_images, axis=0))
    # plt.show()
