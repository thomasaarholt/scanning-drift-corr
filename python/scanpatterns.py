from ase.cluster.cubic import FaceCenteredCubic
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
import hyperspy.api as hs
import math

Signal2D = hs.signals.Signal2D
Gaussian2D = hs.model.components2D.Gaussian2D
import numpy as np

def test_model_cell(angle=0, extra_space=2):
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [3, 4, 5]
    lc = 3.61000
    cell = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc)
    cell[46].number = 50

    m = create_model(cell, extra_space=extra_space, pbc=False)
    m2 = rotate_model(m, cell, angle)
    return m2, cell

def create_model(cell, extra_space=0, pbc=True, pixel_size: "Å" =0.1):
    XLEN, YLEN = ((cell.cell.diagonal()[:2] + 2*extra_space) // pixel_size).astype(int)
    ax0 = {
        'name':'y',
        'size': YLEN,
        'offset':-extra_space,
        'scale':pixel_size, 
        'units':'Å',
    }
    ax1 = {
        'name':'x', 
        'size': XLEN, 
        'offset':-extra_space,
        'scale':pixel_size, 
        'units':'Å',
    }

    axes = [ax0, ax1]
    s = Signal2D(np.zeros((YLEN, XLEN)), axes=axes)

    m = s.create_model()


    if pbc:
        a = [-1, 0, 1]
        merge = [list(zip(x,a)) for x in permutations(a,len(a))]
        l = []
        for entry in merge:
            l += entry
        shifts = set(l)
    else:
        shifts = [0]

    cell_center = np.array([(ax.high_value + ax.scale + ax.low_value)/2 for ax in s.axes_manager.signal_axes])
    diagonal_radius = np.array([ax.high_value for ax in s.axes_manager.signal_axes])
    
    sigma = 0.4
    for atom in cell:
        for offset in shifts:
            xyposition = atom.position[:2] + cell.cell.diagonal()[:2] * offset
            if np.abs(np.linalg.norm(xyposition - cell_center)) > np.linalg.norm(cell_center - diagonal_radius) + 1:
                continue
            A = atom.number**2
            x, y = xyposition
            g = Gaussian2D(A, sigma, sigma, x, y )
            m.append(g)
    return m

def rotation_matrix(deg):
    c = np.cos(np.deg2rad(deg))
    s = np.sin(np.deg2rad(deg))
    return np.array([[c, -s],[s, c]])
    
def rotate_model(m, cell, angle=90):
    center = cell.cell.diagonal()[:2] / 2
    cx, cy = center

    for comp in m:
        comp_center = [comp.centre_x.value, comp.centre_y.value]
        comp.centre_x.value, comp.centre_y.value = rotation_matrix(angle) @ (comp_center - center) + center
    return m

def probe_positions(m, drift_vector, scale=None, scannoise=True, xlen=None, ylen=None, start: "(x,y)"=None):
    if xlen == None or ylen == None:
        xlen, ylen = m.axes_manager.signal_shape
    if scale == None:
        scale = m.axes_manager[-1].scale
    if start == None:
        start = [ax.offset for ax in m.axes_manager.signal_axes]
    X = np.zeros((ylen, xlen))
    Y = np.zeros((ylen, xlen))
    
    xdrift = 0
    ydrift = 0

    if scannoise == True:
        xnoise = np.random.random() * 0.1
    elif scannoise == False:
        xnoise = 0
    else:
        xnoise = scannoise
    
    for yi in range(ylen):
        xnoise = np.random.random() * 0.1 # flyback noise

        for xi in range(xlen):
            xdrift -= drift_vector[0]
            ydrift -= drift_vector[1]

            X[yi, xi] = xi*scale + start[0] + xdrift*scale + xnoise
            Y[yi, xi] = yi*scale + start[1] + ydrift*scale
    return X, Y

def probe_positions_sinus(m, drift_vector, scale=None, period=1000, strength=1, xlen=None, ylen=None, start: "(x,y)"=None):
    if xlen == None or ylen == None:
        xlen, ylen = m.axes_manager.signal_shape
    if scale == None:
        scale = m.axes_manager[-1].scale
    if start == None:
        start = [ax.offset for ax in m.axes_manager.signal_axes]
    X = np.zeros((ylen, xlen))
    Y = np.zeros((ylen, xlen))
    
    xdrift = 0
    ydrift = 0

    i = 0
    for yi in range(ylen):

        for xi in range(xlen):
            sinus = strength*math.sin(2*np.pi*i/period)
            i += 1

            xdrift -= drift_vector[0]
            ydrift -= drift_vector[1]

            X[yi, xi] = xi*scale + start[0] + xdrift*scale + sinus*scale
            Y[yi, xi] = yi*scale + start[1] + ydrift*scale
    return X, Y


def intensity_nd(m, X, Y):
    intensity = np.zeros(X.shape)
    for comp in m:
        if comp.active:
            intensity += comp.function_nd(X, Y)
    return intensity

def image_from_probe_positions(m, XY):
    I = intensity_nd(m, *XY)
    try:
        return m.signal._deepcopy_with_new_data(I)
    except:
        return m._deepcopy_with_new_data(I)

def drift_image(m, drift_vector, scale=None):
    XY = probe_positions(m=m, drift_vector=drift_vector, scale=scale)
    s = image_from_probe_positions(m, XY)
    return s

def drift_image_sinus(m, drift_vector, scale=None, period=1000, strength=1 ):
    XY = probe_positions_sinus(m=m, drift_vector=drift_vector, scale=scale, period=period, strength=strength)
    s = image_from_probe_positions(m, XY)
    return s