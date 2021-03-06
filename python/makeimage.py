import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def SPmakeImage(ss, indImage=0, debug=False):
    indLines = np.ones(ss.scanLines.shape[2], dtype=bool)

    # Unclear if arange in t = should be 0 to +0 or not
    # Edit: Apparently, it needs to be 2 to +2 in order to match matlab
    tt = np.arange(2, ss.scanLines.shape[2] + 2)

    xInd = (
        ss.scanOr[indImage, 0, indLines][:, None] + tt * ss.scanDir[indImage, 0]
    ).flatten()
    yInd = (
        ss.scanOr[indImage, 1, indLines][:, None] + tt * ss.scanDir[indImage, 1]
    ).flatten()
    # scanOr[indLines, 1][:, None] + (t*scanDir[1])[None, :]

    xInd2 = np.clip(xInd, 0, ss.imageSize[0] - 2)  # end with -1)?
    yInd2 = np.clip(yInd, 0, ss.imageSize[1] - 2)

    # Bilinear interpolation on the image in a 2x2 pixel fashion¨
    # xAll and yAll are the coordinates of each pixel, w1 the intensity
    xIndF = np.floor(xInd2).astype(int)
    yIndF = np.floor(yInd2).astype(int)
    xAll = np.array([xIndF, xIndF + 1, xIndF, xIndF + 1])
    with open("results.txt", "a") as f:
        f.write(str(xAll.max(1) + 1))
        f.write("\n")
    # print(xAll.max(1)+1)
    yAll = np.array([yIndF, yIndF, yIndF + 1, yIndF + 1])
    dx = xInd - xIndF
    dy = yInd - yIndF
    w1 = np.array([(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy])

    if debug:
        breakpoint()
    # Get the 2D indices of the bilinear interpolation
    indAll = np.ravel_multi_index((xAll, yAll), ss.imageSize)
    sL = ss.scanLines[indImage, :, indLines].T

    weights = (w1 * sL.flatten()).flatten()
    acc = np.bincount(
        indAll.flatten(), weights=weights, minlength=np.prod(ss.imageSize)
    )

    weights = w1.flatten()
    acc2 = np.bincount(
        indAll.flatten(), weights=weights, minlength=np.prod(ss.imageSize)
    )

    sig = np.reshape(acc, ss.imageSize)
    count = np.reshape(acc2, ss.imageSize)

    r = np.maximum(np.ceil(ss.KDEsigma * 3), 5.0)
    # Just do gaussian smoothing instead?
    sig = gaussian_filter(sig, ss.KDEsigma)
    count = gaussian_filter(count, ss.KDEsigma)

    sub = count > 0
    sig[sub] = sig[sub] / count[sub]
    ss.imageTransform[indImage] = sig
    # plt.figure()
    # plt.imshow(sig)
    # plt.show()
    bound = (count == 0).astype(int)
    bound[[0, -1]] = 1
    bound[:, [0, -1]] = 1
    euclid = distance_transform_edt(1 - bound)
    euclidian_min = np.minimum(euclid / ss.edgeWidth, 1)
    ss.imageDensity[indImage] = np.sin(euclidian_min * np.pi / 2) ** 2
    return ss
