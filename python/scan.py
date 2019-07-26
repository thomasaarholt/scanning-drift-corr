import numpy as np
import pickle
import matplotlib.pyplot as plt

from dataclasses import dataclass

from makeimage_new import SPmakeImage
from sMergeClass import sMergeClass

from tqdm.auto import tqdm


def gaussian_filter_fft(img, kernel):
    fourier = np.fft.fft2(img) * np.fft.fft2(kernel, img.shape)
    return np.fft.ifft2(fourier).real


def gaussian_filter_opencv(img, n, std):
    return cv2.GaussianBlur(src=img, ksize=(n, n), sigmaX=std, sigmaY=std)


def hanningLocal(N):
    # Should this be 1, N+1? Or 0, N?
    Dim1Window = np.sin(np.pi * np.arange(1, N + 1) / (N + 1)) ** 2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def gaussian_kernel(n, std, normalised=False):
    """
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1."""
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= 2 * np.pi * (std ** 2)
    return gaussian2D


def linear_drift(ss):
    # Problems in here
    for a0 in tqdm(range(len(ss.linearSearch))):
        for a1 in tqdm(range(len(ss.linearSearch))):
            xyShift = [ss.inds * ss.xDrift[a0, a1], ss.inds * ss.yDrift[a0, a1]]

            # here, scanOr[0,0] is 1 value behind the Matlab version
            # scanOr[-1, -1] is 65, in both versions
            ss.scanOr[:2] = ss.scanOr[:2] + np.tile(xyShift, [2, 1, 1])

            ss = SPmakeImage(ss, 0)
            ss = SPmakeImage(ss, 1)

            m = np.fft.fft2(ss.w2 * ss.imageTransform[0]) * np.conj(
                ss.w2 * ss.imageTransform[1]
            )
            Icorr = np.fft.ifft2(np.sqrt(abs(m)) * np.exp(1j * np.angle(m))).real
            ss.linearSearchScore1[a0, a1] = np.max(Icorr)
            ss.scanOr[:2] -= np.tile(xyShift, [2, 1, 1])
    return ss


def SPmerge01(data, scanAngles):
    with open("test.txt", "w+") as f:
        f.write("")
    skip = True
    import numpy as np

    data = np.array(data)
    scanAngles = np.array(scanAngles)

    ss = sMergeClass(data, scanAngles)
    ss.linearSearch = ss.linearSearch * ss.scanLines.shape[1]
    ss.yDrift, ss.xDrift = np.meshgrid(ss.linearSearch, ss.linearSearch)
    ss.linearSearchScore1 = np.zeros((len(ss.linearSearch), len(ss.linearSearch)))
    ss.inds = np.linspace(-0.5, 0.5, ss.scanLines.shape[1]).T

    _, xpx, ypx = ss.shape

    TwoDHanningWindow = hanningLocal(xpx) * hanningLocal(ypx).T
    padamount = ss.imageSize - ss.shape[1:]
    pada, padb = padamount
    paddedarray = np.pad(
        TwoDHanningWindow, ((0, pada), (0, padb)), mode="constant", constant_values=0
    )

    ss.w2 = np.roll(paddedarray, np.round(padamount / 2).astype(int), axis=(0, 1))

    # First linear

    if not skip:
        with open("test.txt", "a") as f:
            f.write("Linear1\n")
        ss = linear_drift(ss)

        # Second linear
        ind = np.argmax(ss.linearSearchScore1.flatten())
        xInd, yInd = np.unravel_index(ind, ss.linearSearchScore1.shape)

        step = ss.linearSearch[1] - ss.linearSearch[0]
        xRefine = (
            ss.linearSearch[xInd] + np.linspace(-0.5, 0.5, len(ss.linearSearch)) * step
        )
        yRefine = (
            ss.linearSearch[yInd] + np.linspace(-0.5, 0.5, len(ss.linearSearch)) * step
        )

        ss.yDrift, ss.xDrift = np.meshgrid(yRefine, xRefine)
        ss.linearSearchScore2 = np.zeros((len(ss.linearSearch), len(ss.linearSearch)))
        with open("test.txt", "a") as f:
            f.write("Linear2\n")
        ss = linear_drift(ss)
        with open("savedstate.pickle", "wb") as handle:
            pickle.dump({"s": ss}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("savedstate.pickle", "rb") as handle:
            ss = pickle.load(handle)["s"]

    ind = np.argmax(ss.linearSearchScore1.flatten())
    xInd, yInd = np.unravel_index(ind, ss.linearSearchScore2.shape)
    ss.xyLinearDrift = np.array([ss.xDrift[xInd, 0], ss.yDrift[yInd, 0]])

    # Apply linear shift
    # combine

    xyShift = (ss.inds[:, None] * ss.xyLinearDrift).T
    with open("test.txt", "a") as f:
        f.write("Line1\n")
    for a0 in range(ss.numImages):
        ss.scanOr[a0] = ss.scanOr[a0] + xyShift
        ss = SPmakeImage(ss, a0, debug=False)

    with open("test.txt", "a") as f:
        f.write("Line2\n")
    dxy = np.zeros((ss.numImages, 2))
    G1 = np.fft.fft2(ss.w2 * ss.imageTransform[0])


    for a0 in range(1, ss.numImages):
        breakpoint()
        plt.figure()
        plt.imshow(ss.w2 * ss.imageTransform[a0])
        plt.show()
        G2 = np.fft.fft2(ss.w2 * ss.imageTransform[a0])
        m = G1 * np.conj(G2)
        m2 = np.sqrt(abs(m))
        euler = np.exp(1j * np.angle(m))
        Icorr = np.fft.ifft2(m2 * euler).real
        ind = np.argmax(Icorr.flatten())

        dx, dy = np.unravel_index(ind, Icorr.shape)

        # Might need (Icorr.shape[0] - 1)/2
        dx2 = (dx - 1 + Icorr.shape[0] / 2) % (Icorr.shape[0] - Icorr.shape[0] / 2)
        dy2 = (dy - 1 + Icorr.shape[1] / 2) % (Icorr.shape[1] - Icorr.shape[1] / 2)
        dxy[a0] = dxy[a0 - 1] + np.array([dx2, dy2])
        G1 = G2

    dxy[:, 0] -= np.mean(dxy[:, 0])
    dxy[:, 1] -= np.mean(dxy[:, 1])
    # Apply alignments and regenerate images

    ss.scanOr = (ss.scanOr.T + dxy.T).T
    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # ax1.imshow(ss.imageTransform[0])
    # ax2.imshow(ss.imageTransform[1])
    # ax3.imshow(ss.imageTransform[2])
    # plt.show()

    for a0 in range(ss.numImages):
        ss = SPmakeImage(ss, a0, debug=True)
        # if a0 == 1:
        #     ss = SPmakeImage(ss, a0, debug=False)
        # else:
        #     ss = SPmakeImage(ss, a0, debug=False)

    ss.ref = np.round(ss.imageSize / 2).astype(int)
    imagePlot = np.mean(ss.imageTransform, axis=0)
    dens = np.prod(ss.imageDensity, 0)  # This comes out as zero
    # Scale intensity of image
    mask = dens > 0.5
    imagePlot = imagePlot - np.mean(imagePlot[mask])
    imagePlot = imagePlot / np.sqrt(np.mean(imagePlot[mask] ** 2))

    plt.figure()
    plt.imshow(ss.imageTransform[0])
    plt.show()

    plt.figure()
    plt.imshow(imagePlot)
    plt.show()
    return ss
