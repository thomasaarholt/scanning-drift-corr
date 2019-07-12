import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde
import scipy.stats as st
from dataclasses import dataclass
from scipy.ndimage.morphology import distance_transform_edt

from tqdm.auto import tqdm


def gaussian_filter_fft(img, kernel):
    fourier = np.fft.fft2(img) * np.fft.fft2(kernel, img.shape)
    return np.fft.ifft2(fourier).real


def gaussian_filter_opencv(img, n, std):
    return cv2.GaussianBlur(src=img, ksize=(n, n), sigmaX=std, sigmaY=std)


def rot_matrix(theta):
    'Thanks Hamish'
    return np.array([
        [cosd(theta), -sind(theta)],
        [sind(theta), cosd(theta)]
    ])


def sind(deg):
    return np.sin(np.deg2rad(deg))


def cosd(deg):
    return np.cos(np.deg2rad(deg))


def hanningLocal(N):
    Dim1Window = np.sin(np.pi*np.arange(1, N+1)/(N+1))**2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D


@dataclass
class sMergeClass:
    '''Python dataclass containing the registration data
    scanLines and scanAngles should both be lists or numpy arrays
    '''
    scanLines: np.ndarray
    scanAngles: np.ndarray
    paddingScale: float = (1+1/4)
    KDEsigma: float = 1/2  # - Smoothing between pixels for KDE.
    # - size of edge blending relative to input images.
    edgeWidth: float = 1/128
    # Initial linear search vector, relative to image size.
    linearSearch: np.ndarray = np.linspace(-0.02, 0.02, 1+2*2)

    def __post_init__(self):
        shape = np.array(self.scanLines.shape)
        self.shape = shape

        self.imageSize = np.round(shape[1:]*self.paddingScale/4).astype(int)*4
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
            xy = np.stack([np.arange(0, shape[1]), np.ones(shape[1])])
            xy -= (shape[1:]/2)[:, None]

            xy = rot_matrix(theta) @ xy

            xy += (self.imageSize/2)[:, None]
            xy -= (xy[:, 0] % 1)[:, None]
            self.scanOr[a0] = xy
            self.scanDir[a0] = [cosd(theta + 90), sind(theta + 90)]


def linear_drift(sMerge):
    for a0 in tqdm(range(len(sMerge.linearSearch))):
        for a1 in tqdm(range(len(sMerge.linearSearch))):
            xyShift = [sMerge.inds*sMerge.xDrift[a0, a1],
                       sMerge.inds*sMerge.yDrift[a0, a1]]

            sMerge.scanOr[:2] += np.tile(xyShift, [2, 1, 1])
            sMerge = SPmakeImage(sMerge, 0)
            sMerge = SPmakeImage(sMerge, 1)

            m = np.fft.fft2(
                sMerge.w2*sMerge.imageTransform[0]
            ) * np.conj(
                sMerge.w2*sMerge.imageTransform[1])
            Icorr = np.fft.ifft2(np.sqrt(abs(m))*np.exp(1j*np.angle(m))).real
            sMerge.linearSearchScore1[a0, a1] = np.max(Icorr)
            sMerge.scanOr[:2] -= np.tile(xyShift, [2, 1, 1])
    return sMerge


def SPmerge01(data, scanAngles):
    skip = False
    import numpy as np

    data = np.array(data)
    scanAngles = np.array(scanAngles)

    sMerge = sMergeClass(data, scanAngles)

    sMerge.linearSearch = sMerge.linearSearch * sMerge.scanLines.shape[1]
    sMerge.yDrift, sMerge.xDrift = np.meshgrid(
        sMerge.linearSearch, sMerge.linearSearch)
    sMerge.linearSearchScore1 = np.zeros(
        (len(sMerge.linearSearch), len(sMerge.linearSearch)))
    sMerge.inds = np.linspace(-0.5, 0.5, sMerge.scanLines.shape[1]).T

    _, xpx, ypx = sMerge.shape

    TwoDHanningWindow = hanningLocal(xpx)*hanningLocal(ypx).T
    padamount = sMerge.imageSize - sMerge.shape[1:]
    pada, padb = padamount
    paddedarray = np.pad(
        TwoDHanningWindow,
        ((0, pada), (0, padb)),
        mode='constant', constant_values=0)

    sMerge.w2 = np.roll(paddedarray, np.round(padamount/2).astype(int))

# First linear

    if not skip:
        sMerge = linear_drift(sMerge)

    # Second linear
        ind = np.argmax(sMerge.linearSearchScore1.flatten())
        xInd, yInd = np.unravel_index(ind, sMerge.linearSearchScore1.shape)

        step = sMerge.linearSearch[1] - sMerge.linearSearch[0]
        xRefine = sMerge.linearSearch[xInd] + \
            np.linspace(-0.5, 0.5, len(sMerge.linearSearch))*step
        yRefine = sMerge.linearSearch[yInd] + \
            np.linspace(-0.5, 0.5, len(sMerge.linearSearch))*step

        sMerge.yDrift, sMerge.xDrift = np.meshgrid(yRefine, xRefine)
        sMerge.linearSearchScore2 = np.zeros(
            (len(sMerge.linearSearch), len(sMerge.linearSearch)))

        sMerge = linear_drift(sMerge)

    else:
        with open('savedstate.pickle', 'rb') as handle:
            sMerge = pickle.load(handle)['sMerge']

    ind = np.argmax(sMerge.linearSearchScore1.flatten())
    xInd, yInd = np.unravel_index(ind, sMerge.linearSearchScore2.shape)
    sMerge.xyLinearDrift = np.array(
        [sMerge.xDrift[xInd, 0], sMerge.yDrift[yInd, 0]])

    # Apply linear shift
    xyShift = [sMerge.inds*sMerge.xyLinearDrift[0],
               sMerge.inds*sMerge.xyLinearDrift[1]]

    for a0 in range(sMerge.numImages):
        sMerge.scanOr[a0] = sMerge.scanOr[a0] + xyShift
    sMerge = SPmakeImage(sMerge, a0)
    dxy = np.zeros((sMerge.numImages, 2))

    G1 = np.fft.fft2(sMerge.w2*sMerge.imageTransform[0])
    for a0 in range(1, sMerge.numImages):
        G2 = np.fft.fft2(sMerge.w2*sMerge.imageTransform[a0])
        m = G1*np.conj(G2)
        Icorr = np.fft.ifft2(np.sqrt(abs(m))*np.exp(1j*np.angle(m))).real
        ind = np.argmax(Icorr.flatten())

        dx, dy = np.unravel_index(ind, Icorr.shape)

        # Might need (Icorr.shape[0] - 1)/2
        dx = dx - 1 + Icorr.shape[0]/2 % Icorr.shape[0] - Icorr.shape[0]/2
        dy = dy - 1 + Icorr.shape[1]/2 % Icorr.shape[1] - Icorr.shape[1]/2
        dxy[a0] = dxy[a0-1] + np.array([dx, dy])
        G1 = G2

    dxy[:, 0] -= np.mean(dxy[:, 0])
    dxy[:, 1] -= np.mean(dxy[:, 1])
    # Apply alignments and regenerate images
    for a0 in range(sMerge.numImages):
        # error on a0 = 1

        sMerge.scanOr[a0, 0] += dxy[a0, 0]
        sMerge.scanOr[a0, 1] += dxy[a0, 1]
        if a0 == 1:
            sMerge = SPmakeImage(sMerge, a0, debug=False)
        else:
            sMerge = SPmakeImage(sMerge, a0)

    sMerge.ref = np.round(sMerge.imageSize/2).astype(int)
    imagePlot = np.mean(sMerge.imageTransform, axis=0)
    dens = np.prod(sMerge.imageDensity, 0)
    # Scale intensity of image
    mask = dens > 0.5
    imagePlot = imagePlot - np.mean(imagePlot[mask])
    imagePlot = imagePlot / np.sqrt(np.mean(imagePlot[mask]**2))

    plt.figure()
    plt.imshow(imagePlot)
    plt.show()
    return sMerge


def SPmakeImage(sMerge, indImage=0, debug=False):
    indLines = np.ones(sMerge.scanLines.shape[2], dtype=bool)

    # Unclear if arange in t = should be 0 to +0 or not
    t = np.tile(
        np.arange(1, sMerge.scanLines.shape[2]+1), reps=[indLines.sum(), 1])
    xtile = sMerge.scanOr[indImage, 0, indLines]
    ytile = sMerge.scanOr[indImage, 1, indLines]
    x0 = np.tile(xtile, reps=[sMerge.scanLines.shape[2], 1])
    y0 = np.tile(ytile, reps=[sMerge.scanLines.shape[2], 1])

    # Might need to remove .T on this line only
    xInd = x0.T.flatten() + t.flatten()*sMerge.scanDir[indImage, 0]
    yInd = y0.T.flatten() + t.flatten()*sMerge.scanDir[indImage, 1]

    xInd2 = np.clip(xInd, 0, sMerge.imageSize[0]-1)
    yInd2 = np.clip(yInd, 0, sMerge.imageSize[1]-1)

    # Bilinear interpolation on the image in a 2x2 pixel fashion¨
    # xAll and yAll are the coordinates of each pixel, w1 the intensity
    xIndF = np.floor(xInd2).astype(int)
    yIndF = np.floor(yInd2).astype(int)
    xAll = np.array([xIndF, xIndF+1, xIndF, xIndF+1])
    yAll = np.array([yIndF, yIndF, yIndF+1, yIndF+1])
    dx = xInd-xIndF
    dy = yInd-yIndF
    w1 = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])

    if debug:
        breakpoint()
    # Get the 2D indices of the bilinear interpolation
    indAll = np.ravel_multi_index((xAll, yAll), sMerge.imageSize)
    sL = sMerge.scanLines[indImage, :, indLines].T

    weights = (w1*sL.flatten()).flatten()
    acc = np.bincount(
        indAll.flatten(),
        weights=weights,
        minlength=np.prod(sMerge.imageSize))

    weights = w1.flatten()
    acc2 = np.bincount(
        indAll.flatten(), weights=weights, minlength=np.prod(sMerge.imageSize))

    sig = np.reshape(acc, sMerge.imageSize)
    count = np.reshape(acc2, sMerge.imageSize)
    # plt.figure()
    # plt.imshow(sig)
    # plt.show()

    r = np.maximum(np.ceil(sMerge.KDEsigma*3), 5.0)
    kern = gkern(2*r+1, sMerge.KDEsigma)
    # Just do gaussian smoothing instead?
    sig = gaussian_filter(sig, sMerge.KDEsigma)
    count = gaussian_filter(count, sMerge.KDEsigma)

    sub = count > 0
    sig[sub] = sig[sub] / count[sub]
    sMerge.imageTransform[indImage] = sig

    bound = (count == 0).astype(int)
    bound[[0, -1]] = 1
    bound[:, [0, -1]] = 1
    euclid = distance_transform_edt(1-bound)
    euclidian_min = np.minimum(
        euclid/sMerge.edgeWidth, 1)
    sMerge.imageDensity[indImage] = np.sin(euclidian_min*np.pi/2)**2
    return sMerge

#     % Apply KDE
#     r = max(ceil(sMerge.sMerge.KDEsigma*3),5);
#     sm = fspecial('gaussian',2*r+1,sMerge.sMerge.KDEsigma);
#     sm = sm / sum(sm(:));
#     sig = conv2(sig,sm,'same');
#     count = conv2(count,sm,'same');
#     sub = count > 0;
#     sig(sub) = sig[sub] / count[sub];
#     sMerge.sMerge.imageTransform(:,:,indImage) = sig;
