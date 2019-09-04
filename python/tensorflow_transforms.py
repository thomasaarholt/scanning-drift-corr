import numpy as np
import tensorflow as tf
from tensorflow.math import conj, angle, sqrt, real, reduce_mean as mean
from tensorflow.signal import ifft2d as ifft2, fft2d as fft2, rfft2d, irfft2d

from time import time


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    logging.getLogger("tensorflow").setLevel(level)


def hybrid_correlation_tf(img1_fft, img2_fft):
    """Unsure if this actually the hybrid correlation Ophus refers to.
    Works on images already in fourier space.
    """
    m = img1_fft * conj(img2_fft)
    M = sqrt(tf.abs(m))
    magnitude = tf.complex(M, M * 0.0)
    theta = angle(m)
    euler = tf.exp(tf.complex(theta * 0.0, theta))
    D = magnitude * euler
    Icorr = real(ifft2(D))
    return Icorr


def phase_correlation_tf(img1_fft, img2_fft):
    "Perform phase correlation on images that are already in fourier space"
    C = img1_fft * conj(img2_fft)
    D = tf.abs(C)
    return real(ifft2(C / tf.complex(D, D * 0.0)))


def cross_correlation_tf_old(im1, im2):
    "Perform cross correlation on images in real space"
    # get rid of the averages, otherwise the results are not good
    im1 -= mean(im1)
    im2 -= mean(im2)

    # calculate the correlation image; note the flipping of the images
    return fftconv(im1, im2[::-1, ::-1], mode="same")


def cross_correlation_tf(A, B):
    A = A - tf.math.reduce_mean(A)
    B = B - tf.math.reduce_mean(B)
    R = real(ifft2(fft2(A) * fft2(B[..., ::-1, ::-1])))
    return R


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = tf.shape(arr)[-2:]
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0] : endind[0], startind[1] : endind[1]]


def fftconv(in1, in2, mode="full"):
    # Reorder channels to come second (needed for fft)
    # in1 = tf.transpose(in1, perm=[0, 3, 1, 2])
    # in2 = tf.transpose(in2, perm=[0, 3, 1, 2])

    # Extract shapes
    s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
    s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
    shape = s1 + s2 - 1

    # Compute convolution in fourier space
    sp1 = rfft2d(in1, shape)
    sp2 = rfft2d(in2, shape)
    ret = irfft2d(sp1 * sp2, shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'.")

    # Reorder channels to last
    return cropped
