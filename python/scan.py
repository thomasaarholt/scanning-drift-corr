import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def sind(deg):
    return np.sin(np.deg2rad(deg))


def cosd(deg):
    return np.cos(np.deg2rad(deg))


def hanningLocal(N):
    N = N
    Dim1Window = np.sin(np.pi*np.arange(1, N+1)/(N+1))**2
    Dim2Window = Dim1Window[:, None]
    return Dim2Window


def SPmerge01(data, scanAngles):
    import numpy as np
    # 1.5;    # - padding amount for scaling of the output.
    paddingScale = (1+1/4)
    KDEsigma = 1/2  # - Smoothing between pixels for KDE.
    edgeWidth = 1/128  # - size of edge blending relative to input images.
    # Initial linear search vector, relative to image size.
    linearSearch = np.linspace(-0.02, 0.02, 1+2*2)

    # I have changed the convention here so that the number of images is the first axis, not the last
    # I am only supporting the case where the image input is a list / array of images

    data = np.array(data)
    scanAngles = np.array(scanAngles)
    shape = np.array(data.shape)
    imageSize = np.round(shape[1:]*paddingScale/4).astype(int)*4
    numImages = len(scanAngles)

    scanLines = np.zeros(shape)
    scanOr = np.zeros([numImages, 2, shape[1]])

    scanDir = np.zeros([numImages, 2])

    imageTransform = scanLines.copy()
    imageDensity = scanLines.copy()

    scanLines[:] = data

    for a0 in range(numImages):

        xy = np.column_stack([np.arange(1, shape[1]+1), np.ones(shape[1])])
        xy[:, 0] -= shape[1]/2
        xy[:, 1] -= shape[2]/2

        xy = np.array([
            xy[:, 0]*cosd(scanAngles[a0])
            - xy[:, 1]*sind(scanAngles[a0]),
            xy[:, 1]*cosd(scanAngles[a0])
            + xy[:, 0]*sind(scanAngles[a0]),
        ])

        xy[0] += imageSize[0]/2
        xy[1] += imageSize[1]/2
        xy[0] -= xy[0, 0] % 1
        xy[1] -= xy[1, 0] % 1
        scanOr[a0] = xy

        scanDir[a0] = [cosd(scanAngles[a0] + 90), sind(scanAngles[a0] + 90)]

    linearSearch = linearSearch * scanLines.shape[1]
    yDrift, xDrift = np.meshgrid(linearSearch, linearSearch)
    linearSearchScore1 = np.zeros(len(linearSearch))
    inds = np.linspace(-0.5, 0.5, scanLines.shape[1]).T

    N = scanLines.shape

    TwoDHanningWindow = hanningLocal(N[1])*hanningLocal(N[2]).T
    padamount = imageSize - N[1:]
    pada, padb = padamount
    paddedarray = np.pad(TwoDHanningWindow, ((0, pada),
                                             (0, padb)), mode='constant', constant_values=0)

    w2 = np.roll(paddedarray, np.round(padamount/2).astype(int))

    for a0 in range(len(linearSearch)):
        for a1 in range(len(linearSearch)):
            xyShift = [inds*xDrift[a0, a1], inds*yDrift[a0, a1]]

            scanOr[:2] += np.tile(xyShift, [2, 1, 1])
            SPmakeImage(scanLines, imageSize, KDEsigma, scanOr, scanDir, 0)
            break
        break


def SPmakeImage(scanLines, imageSize, KDEsigma, scanOr, scanDir, indImage=0):
        # SPmakeImage

    indLines = np.ones(scanLines.shape[2], dtype=int)

    # Remove the plus 1s below
    t = np.tile(
        np.arange(0+1, scanLines.shape[2]+1), [indLines.sum(dtype=int), 1])
    x0 = np.tile(scanOr[indImage, 0, indLines.astype(bool)], [
                 1, scanLines.shape[2]])
    y0 = np.tile(scanOr[indImage, 1, indLines.astype(bool)], [
                 1, scanLines.shape[2]])
    # print(scanDir.shape)
    xInd = x0.flatten() + t.flatten()*scanDir[indImage, 0]
    yInd = y0.flatten() + t.flatten()*scanDir[indImage, 1]

    xInd2 = np.maximum(np.minimum(xInd, imageSize[0]-1), 1)
    yInd2 = np.maximum(np.minimum(yInd, imageSize[1]-1), 1)

    xIndF = np.floor(xInd2).astype(int)
    yIndF = np.floor(yInd2).astype(int)
    xAll = np.array([xIndF, xIndF+1, xIndF, xIndF+1])
    yAll = np.array([yIndF, yIndF, yIndF+1, yIndF+1])
    dx = xInd-xIndF
    dy = yInd-yIndF
    w1 = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])

    # print(xAll.dtype)
    # print(imageSize.dtype)

   # breakpoint()

    indAll = np.ravel_multi_index((xAll, yAll), imageSize)
    sL = scanLines[indImage, :, indLines.astype(bool)].T
    #indAll = sub2ind(sMerge.imageSize,xAll,yAll);

    weights = np.array([w1[0]*sL.flatten(), w1[1]*sL.flatten(),
                        w1[2]*sL.flatten(), w1[3]*sL.flatten()])
    breakpoint()
    accumarray = np.bincount(
        indAll.flatten(), weights=weights.flatten(), minlength=np.prod(imageSize))
    sig = np.reshape(accumarray, imageSize)

    weights2 = np.array([w1[0], w1[1], w1[2], w1[3]])
    accumarray2 = np.bincount(
        indAll.flatten(), weights=weights2.flatten(), minlength=np.prod(imageSize))

    count = np.reshape(accumarray2, imageSize)

    r = np.maximum(np.ceil(KDEsigma*3), 5.0)
    plt.figure()
    plt.imshow(sig)
    plt.show()
    # gaussian_filter()


#     % Apply KDE
#     r = max(ceil(sMerge.KDEsigma*3),5);
#     sm = fspecial('gaussian',2*r+1,sMerge.KDEsigma);
#     sm = sm / sum(sm(:));
#     sig = conv2(sig,sm,'same');
#     count = conv2(count,sm,'same');
#     sub = count > 0;
#     sig(sub) = sig[sub] / count[sub];
#     sMerge.imageTransform(:,:,indImage) = sig;
