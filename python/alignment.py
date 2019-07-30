import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve
from time import time


def cross_correlation(img1_fft, img2_fft):
    m = img1_fft * np.conj(img2_fft)

    magnitude = np.sqrt(abs(m))
    euler = np.exp(1j * np.angle(m))
    Icorr = np.fft.ifft2(magnitude * euler).real
    return Icorr


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

    correlations = [
        fftconvolve(img1, img2[::-1, ::-1], mode="same")
        for img1, img2 in zip(first_imgs, last_imgs)
    ]

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    img1, img2 = rotated_images
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

    translation = np.array(
        [np.unravel_index(corr.argmax(), corr.shape) for corr in correlations]
    )
    halfshape = new_shape / 2
    translation = translation[::-1] - halfshape
    print(translation)

    # img1, img2 = padded_images
    # print("calc")
    # t = time()
    # corr = fftconvolve(img1, img2[::-1, ::-1], mode="same")
    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # ax1.imshow(img1)
    # ax2.imshow(img2)
    # ax3.imshow(np.abs(corr))
    # plt.show()
    # print(time() - t)

    # shift = np.where(corr == corr.max())
    # shift = (shift[0] % img1.shape[0], shift[1] % img1.shape[1])
    # first_ffts = [np.fft.fft2(weights * img) for img in first_imgs]
    # last_ffts = [np.fft.fft2(weights * img) for img in last_imgs]

    # translation = [
    #     cross_correlation(fft1, fft2) for fft1, fft2 in zip(first_ffts, last_ffts)
    # ]

    # translation = [
    #     np.unravel_index(Icorr.argmax(), Icorr.shape) for Icorr in translation
    # ]

    # translation = np.multiply(translation, -1)

    # # translation = [
    # #     register_translation(weights * img1, weights * img2)[0]
    # #     for img1, img2 in zip(padded_images[:-1], padded_images[1:])
    # # ]

    corrected_images = [rotated_images[0]] + [
        np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)).real
        for img, shift in zip(rotated_images[1:], translation)
    ]

    fig, AX = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    imgs = corrected_images + [
        np.mean(rotated_images, axis=0),
        np.mean(corrected_images, axis=0),
    ]
    for ax, im in zip(AX.flatten(), imgs):
        ax.imshow(im)
    plt.show()
    # np.rotate()
