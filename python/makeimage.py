import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def SPmakeImage(s, indImage=0, debug=False):
    # print('indImage', indImage+1)
    indLines = np.ones(s.scanLines.shape[2], dtype=bool)

    # Unclear if arange in t = should be 0 to +0 or not
    # Edit: Apparently, it needs to be 2 to +2 in order to match matlab
    t = np.tile(
        np.arange(2, s.scanLines.shape[2]+2), reps=[indLines.sum(), 1])
    xtile = s.scanOr[indImage, 0, indLines]
    ytile = s.scanOr[indImage, 1, indLines]
    x0 = np.tile(xtile, reps=[s.scanLines.shape[2], 1])
    y0 = np.tile(ytile, reps=[s.scanLines.shape[2], 1])

    # Might need to remove .T on this line only
    xInd = x0.T.flatten() + t.flatten()*s.scanDir[indImage, 0]
    yInd = y0.T.flatten() + t.flatten()*s.scanDir[indImage, 1]

    xInd2 = np.clip(xInd, 0, s.imageSize[0]-2)  # end with -1)?
    yInd2 = np.clip(yInd, 0, s.imageSize[1]-2)

    # Bilinear interpolation on the image in a 2x2 pixel fashion¨
    # xAll and yAll are the coordinates of each pixel, w1 the intensity
    xIndF = np.floor(xInd2).astype(int)
    yIndF = np.floor(yInd2).astype(int)
    xAll = np.array([xIndF, xIndF+1, xIndF, xIndF+1])
    with open('test.txt', 'a') as f:
        f.write(str(xAll.max(1)+1))
        f.write('\n')
    # print(xAll.max(1)+1)
    yAll = np.array([yIndF, yIndF, yIndF+1, yIndF+1])
    dx = xInd-xIndF
    dy = yInd-yIndF
    w1 = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])

    if debug:
        breakpoint()
    # Get the 2D indices of the bilinear interpolation
    indAll = np.ravel_multi_index((xAll, yAll), s.imageSize)
    sL = s.scanLines[indImage, :, indLines].T

    weights = (w1*sL.flatten()).flatten()
    acc = np.bincount(
        indAll.flatten(),
        weights=weights,
        minlength=np.prod(s.imageSize))

    weights = w1.flatten()
    acc2 = np.bincount(
        indAll.flatten(), weights=weights, minlength=np.prod(s.imageSize))

    sig = np.reshape(acc, s.imageSize)
    count = np.reshape(acc2, s.imageSize)

    r = np.maximum(np.ceil(s.KDEsigma*3), 5.0)
    # Just do gaussian smoothing instead?
    sig = gaussian_filter(sig, s.KDEsigma)
    count = gaussian_filter(count, s.KDEsigma)

    sub = count > 0
    sig[sub] = sig[sub] / count[sub]
    s.imageTransform[indImage] = sig
    # plt.figure()
    # plt.imshow(sig)
    # plt.show()
    bound = (count == 0).astype(int)
    bound[[0, -1]] = 1
    bound[:, [0, -1]] = 1
    euclid = distance_transform_edt(1-bound)
    euclidian_min = np.minimum(
        euclid/s.edgeWidth, 1)
    s.imageDensity[indImage] = np.sin(euclidian_min*np.pi/2)**2
    return s
