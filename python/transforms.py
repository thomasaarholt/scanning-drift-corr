import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

from skimage.transform import AffineTransform, warp
from scipy.ndimage import fourier_shift
from scipy.signal import fftconvolve


def hybrid_correlation(img1_fft, img2_fft):
    "Unsure if this actually is hybrid correlation according to Ophus"
    m = img1_fft * np.conj(img2_fft)

    magnitude = np.sqrt(np.abs(m))
    euler = np.exp(1j * np.angle(m))
    Icorr = np.fft.ifft2(magnitude * euler).real
    return Icorr


def phase_correlation(img1_fft, img2_fft):
    "A and B are FFTs of images"
    C = img1_fft * np.conj(img2_fft)
    return np.fft.ifft2(C / np.abs(C)).real


def cross_correlation(im1, im2):
    # get rid of the averages, otherwise the results are not good
    im1 -= np.mean(im1)
    im2 -= np.mean(im2)

    # calculate the correlation image; note the flipping of the images
    return fftconvolve(im1, im2[::-1, ::-1], mode="same")


def changespace(x, maxval):
    """Shift indices from [0, max] to [-max/2, max/2]
    Handles both single values and numpy shape tuples"""
    x = np.array(x)
    maxval = np.array(maxval)
    half = maxval / 2
    return (x + half) % maxval - half


def phase_translation(corr):
    "Turn phase correlation array into shift x,y"
    shift = np.unravel_index(corr.argmax(), corr.shape)

    shift = changespace(shift, corr.shape)
    return shift


def cross_translation(corr):
    "Turn phase correlation array into shift x,y"
    shift = np.unravel_index(corr.argmax(), corr.shape)
    return shift - np.array(corr.shape) / 2


def pad_images(images, pad_scale=1.25):
    old_shape = np.array(images[0].shape)
    new_shape = old_shape * pad_scale  # approx
    padding = ((new_shape - old_shape) / 2).astype(int)
    padding = [(padding[0], padding[0]), (padding[1], padding[1])]
    padded_images = [
        np.pad(img, padding, "constant", constant_values=0) for img in images
    ]
    new_shape = np.array(padded_images[0].shape)

    window = HanningImage(old_shape[0]) * HanningImage(old_shape[1]).T
    weights = np.pad(window, padding, mode="constant", constant_values=0)
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
    rotation_matrices = [
        AffineTransform(rotation=np.deg2rad(angle)) for angle in angles
    ]
    print(sheares)
    for shear in sheares:
        print(shear)
        shear_matrix = AffineTransform(shear=shear)
        print(shear_matrix.params)
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
    shift_x, shift_y = np.array(img.shape) / 2.0
    tf_shift = AffineTransform(translation=[-shift_y, -shift_x])
    tf_shift_inv = AffineTransform(translation=[shift_y, shift_x])
    breakpoint()
    return weights * warp(
        img,
        (tf_shift + scale_matrix + shear_matrix + rot_matrix + tf_shift_inv).inverse,
        order=0,
    )


def HanningImage(N):
    "Return a 2D hanning window of size NxN "
    han = np.hanning(N)
    return han * han[:, None]


def normalise_max(img):
    "Use to normalise before correlation. Unsure if necessary or a good idea"
    return img / img.max()
