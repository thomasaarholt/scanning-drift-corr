import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde
import scipy.stats as st
from dataclasses import dataclass

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
    N = N
    Dim1Window = np.sin(np.pi*np.arange(1, N+1)/(N+1))**2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


class sMergeClass:
    pass





def SPmerge01(data, scanAngles):
    import numpy as np
    
    sMerge = sMergeClass()

    paddingScale = (1+1/4)
    sMerge.KDEsigma = 1/2  # - Smoothing between pixels for KDE.
    sMerge.edgeWidth = 1/128  # - size of edge blending relative to input images.
    # Initial linear search vector, relative to image size.
    sMerge.linearSearch = np.linspace(-0.02, 0.02, 1+2*2)

    # I have changed the convention here so that the number of images is the first axis, not the last
    # I am only supporting the case where the image input is a list / array of images

    data = np.array(data)
    shape = np.array(data.shape)

    sMerge.scanAngles = np.array(scanAngles)
    sMerge.imageSize = np.round(shape[1:]*paddingScale/4).astype(int)*4
    sMerge.numImages = len(sMerge.scanAngles)
    sMerge.scanLines = np.zeros(shape)
    sMerge.scanOr = np.zeros([sMerge.numImages, 2, shape[1]])
    sMerge.scanDir = np.zeros([sMerge.numImages, 2])

    padded_shape = (sMerge.numImages,) + tuple(sMerge.imageSize)
    sMerge.imageTransform = np.zeros(padded_shape)
    sMerge.imageDensity = np.zeros(padded_shape)

    sMerge.scanLines[:] = data

    for a0 in range(sMerge.numImages):
        theta = sMerge.scanAngles[a0]
        xy = np.stack([np.arange(0, shape[1]), np.ones(shape[1])])
        xy -= (shape[1:]/2)[:, None]

        xy = rot_matrix(theta) @ xy

        xy += (sMerge.imageSize/2)[:, None]
        xy -= (xy[:, 0] % 1)[:, None]
        sMerge.scanOr[a0] = xy
        sMerge.scanDir[a0] = [cosd(theta + 90), sind(theta + 90)]

    sMerge.linearSearch = sMerge.linearSearch * sMerge.scanLines.shape[1]
    yDrift, xDrift = np.meshgrid(sMerge.linearSearch, sMerge.linearSearch)
    sMerge.linearSearchScore1 = np.zeros(len(sMerge.linearSearch))
    inds = np.linspace(-0.5, 0.5, sMerge.scanLines.shape[1]).T

    N = sMerge.scanLines.shape

    TwoDHanningWindow = hanningLocal(N[1])*hanningLocal(N[2]).T
    padamount = sMerge.imageSize - N[1:]
    pada, padb = padamount
    paddedarray = np.pad(
        TwoDHanningWindow,
        ((0, pada), (0, padb)),
        mode='constant', constant_values=0)

    w2 = np.roll(paddedarray, np.round(padamount/2).astype(int))

    for a0 in range(len(sMerge.linearSearch)):
        for a1 in range(len(sMerge.linearSearch)):
            xyShift = [inds*xDrift[a0, a1], inds*yDrift[a0, a1]]

            sMerge.scanOr[:2] += np.tile(xyShift, [2, 1, 1])
            SPmakeImage(sMerge, 0)
            break
        break


def SPmakeImage(sMerge, indImage=0):
    indLines = np.ones(sMerge.scanLines.shape[2], dtype=bool)

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

    xIndF = np.floor(xInd2).astype(int)
    yIndF = np.floor(yInd2).astype(int)
    xAll = np.array([xIndF, xIndF+1, xIndF, xIndF+1])
    yAll = np.array([yIndF, yIndF, yIndF+1, yIndF+1])
    dx = xInd-xIndF
    dy = yInd-yIndF
    w1 = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])

    indAll = np.ravel_multi_index((xAll, yAll), sMerge.imageSize)
    sL = sMerge.scanLines[indImage, :, indLines].T

    weights = (w1*sL.flatten()).flatten()
    acc = np.bincount(
        indAll.flatten(),
        weights=weights,
        minlength=np.prod(sMerge.imageSize))

    acc2 = np.bincount(
        indAll.flatten(), weights=weights, minlength=np.prod(sMerge.imageSize))

    sig = np.reshape(acc, sMerge.imageSize)
    count = np.reshape(acc2, sMerge.imageSize)
    plt.figure()
    plt.imshow(sig)
    plt.show()

    
    r = np.maximum(np.ceil(sMerge.KDEsigma*3), 5.0)
    # Just do gaussian smoothing instead?
    # sig = gaussian_filter(sig, 2)
    # count = gaussian_filter(count, sMerge.KDEsigma)
    kern = gkern(11, 0.5)
    sig = convolve2d(sig, kern, 'same')
    count = convolve2d(count, kern, 'same')

    sub = count > 0
    sig[sub] = sig[sub] / count[sub]
    sMerge.imageTransform[indImage] = sig

    plt.figure()
    plt.imshow(sig)
    plt.show()


#     % Apply KDE
#     r = max(ceil(sMerge.sMerge.KDEsigma*3),5);
#     sm = fspecial('gaussian',2*r+1,sMerge.sMerge.KDEsigma);
#     sm = sm / sum(sm(:));
#     sig = conv2(sig,sm,'same');
#     count = conv2(count,sm,'same');
#     sub = count > 0;
#     sig(sub) = sig[sub] / count[sub];
#     sMerge.sMerge.imageTransform(:,:,indImage) = sig;
