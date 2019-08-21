import matplotlib.pyplot as plt

plt.rcParams["figure.max_open_warning"] = 2000

import h5py
import numpy as np
from numpy.fft import fft2, ifft2

from skimage.transform import rotate
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve

from skimage.transform import AffineTransform, warp


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


def translation_orig(corr):
    translation = np.array(np.unravel_index(corr.argmax(), corr.shape))
    halfshape = np.array(corr.shape) / 2

    translation = (translation + halfshape) % halfshape - halfshape
    return translation


def pad_images(images, pad_scale=1.25):
    pad_scale = 1.25
    old_shape = np.array(images[0].shape)
    new_shape = old_shape * pad_scale  # approx
    padding = ((new_shape - old_shape) / 2).astype(int)
    padding = [(padding[0], padding[0]), (padding[1], padding[1])]
    padded_images = [
        np.pad(img, padding, "constant", constant_values=0) for img in images
    ]
    new_shape = np.array(padded_images[0].shape)

    TwoDHanningWindow = hanningLocal(old_shape[0]) * hanningLocal(old_shape[1]).T
    weights = np.pad(TwoDHanningWindow, padding, mode="constant", constant_values=0)
    return padded_images, weights


def transform(padded_images, angles, weights=1):
    steps = 7

    shape = padded_images[0].shape
    shift_x, shift_y = np.array(shape) / 2.0
    tf_shift = AffineTransform(translation=[-shift_y, -shift_x])
    tf_shift_inv = AffineTransform(translation=[shift_y, shift_x])

    shear_steps = scale_steps = steps  # must be odd

    pix = 1
    # Scale so pix pixels wider with each step
    scale = (shape[1] + pix) / shape[1] - 1
    # Shear by pix pixels
    shear = np.arctan(pix / shape[1])  # maybe scale[0]?

    half_steps = (steps - 1) / 2
    scale_limits = scale * half_steps
    shear_limits = shear * half_steps
    scales = 1 + np.linspace(-scale_limits, scale_limits, scale_steps)
    sheares = np.linspace(-shear_limits, shear_limits, shear_steps)

    data = np.zeros((2, len(sheares), len(scales)) + shape)
    for i, (img, angle) in enumerate(zip(padded_images, angles)):
        tf_rotate = AffineTransform(rotation=np.deg2rad(angle))
        for j, shear in enumerate(sheares):
            tf_shear = AffineTransform(shear=shear)
            for k, scale in enumerate(scales):
                tf_scale = AffineTransform(scale=(1, scale))
                data[i, j, k] = weights * warp(
                    img,
                    (
                        tf_shift + tf_scale + tf_shear + (tf_rotate + tf_shift_inv)
                    ).inverse,
                    order=0,
                )

    return data


def hanningLocal(N):
    # Should this be 1, N+1? Or 0, N?
    Dim1Window = np.sin(np.pi * np.arange(1, N + 1) / (N + 1)) ** 2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def normalise_max(img):
    return img / img.max()


def setsize(fig, px=1000):
    try:
        px = str(px) + "px"
        fig.canvas.layout.width = px
        fig.canvas.layout.height = px
    except:
        pass

def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
            return False
    except:
        return False
    else:  # pragma: no cover
        return True
