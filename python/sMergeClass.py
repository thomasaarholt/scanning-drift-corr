import numpy as np
from dataclasses import dataclass


def sind(deg):
    return np.sin(np.deg2rad(deg))


def cosd(deg):
    return np.cos(np.deg2rad(deg))


def rot_matrix(theta):
    "Thanks Hamish"
    return np.array([[cosd(theta), -sind(theta)], [sind(theta), cosd(theta)]])


@dataclass
class sMergeClass:
    """Python dataclass containing the registration data
    scanLines and scanAngles should both be lists or numpy arrays
    """

    scanLines: np.ndarray
    scanAngles: np.ndarray
    paddingScale: float = (1 + 1 / 4)
    KDEsigma: float = 1 / 2  # - Smoothing between pixels for KDE.
    # - size of edge blending relative to input images.
    edgeWidth: float = 1 / 128
    # Initial linear search vector, relative to image size.
    linearSearch: np.ndarray = np.linspace(-0.02, 0.02, 1 + 2 * 2)

    def __post_init__(self):
        shape = np.array(self.scanLines.shape)
        self.shape = shape

        self.imageSize = np.round(shape[1:] * self.paddingScale / 4).astype(int) * 4
        self.numImages = shape[0]
        self.scanOr = np.zeros([self.numImages, 2, shape[1]])
        self.scanDir = np.zeros([self.numImages, 2])

        padded_shape = (self.numImages,) + tuple(self.imageSize)
        self.imageTransform = np.zeros(padded_shape)
        self.imageDensity = np.zeros(padded_shape)

        self.calculate_directions()

    def calculate_directions(self):
        shape = self.shape
        for a0 in range(self.numImages):
            theta = self.scanAngles[a0]
            # Changing the following line from 0, shape[1] to +1 on both
            # If xy is index value (not difference), then we probably want to
            # reduce them all by one compared to matlab version
            matlab_style = True
            if matlab_style:
                xy = np.stack([np.arange(1, shape[1] + 1), np.ones(shape[1])])
            else:
                xy = np.stack([np.arange(shape[1]), np.zeros(shape[1])])
            xy -= (shape[1:] / 2)[:, None]

            # For two images at 0, 90 deg, the second one will swap the x and y indices and reverse the sign of the first axis.
            xy = rot_matrix(theta) @ xy
            if matlab_style:
                xy -= 1  # Subtract one to get python indices
            xy += (self.imageSize / 2)[:, None]

            xy -= (xy[:, 0] % 1)[:, None]  # I think this is to avoid fractional indices
            self.scanOr[a0] = xy
            self.scanDir[a0] = [cosd(theta + 90), sind(theta + 90)]
