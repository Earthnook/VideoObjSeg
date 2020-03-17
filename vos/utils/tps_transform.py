""" A TPS transofrmation on image in Numpy version
from https://gist.github.com/bgshih/e252ba7148590a381f9c
NOTE: (x, y) coordinates are (W_idx, H_idx) respectively.
And image[:, y, x] gets the pixel you see in plotted image
"""
import numpy as np
import numpy.linalg as nl
from scipy.spatial.distance import pdist, cdist, squareform

def makeT(cp):
    """
    @ Args:
        cp: control points with shape (K, 2)
    @ Returns:
        T: array with shape (K+3, K+3)
    """
    K = cp.shape[0]
    T = np.zeros((K+3, K+3))

    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K+1:, 3:] = cp.T

    R = squareform(pdist(cp, metric= "euclidean"))
    R = R * R
    R[R == 0] = 1 # a trick to make R*ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)

    T[:K, 3:] = R
    return T

def liftPts(p, cp):
    """
    @ Args:
        p: [N, 2], input points
        cp: [K, 2], control points
    @ Returns:
        pLift: [N, (3+K)], lifted input points
    """
    N, K = p.shape[0], cp.shape[0]

    pLift = np.zeros((N, K+3))
    pLift[:,0] = 1
    pLift[:,1:3] = p

    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)

    pLift[:,3:] = R
    return pLift

def get_all_tcps(scps, tcps, shape):
    """ NOTE: source control points must not be collinear with themselves
    @ Args:
        scps: source control points with shape (N, 2) in (x, y) coordinates
        tcps: target control points with shape (N, 2) in (x, y) coordinates, usually jittered from source control points
        shape: a 2-d tuple specifying the image resolution
    @ Returns:
        gts: source as a meshgrid of `shape` and all targets
    """

    # construct T
    T = makeT(scps)

    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([tcps[:,0], np.zeros(3)])
    ytAug = np.concatenate([tcps[:,1], np.zeros(3)])
    cx = nl.solve(T, xtAug) # [K+3]
    cy = nl.solve(T, ytAug)

    # grid for image pixels
    xs = np.arange(shape[1])
    ys = np.arange(shape[0])
    x, y = np.meshgrid(xs, ys)
    xgs, ygs = x.flatten(), y.flatten()
    gps = np.vstack([xgs, ygs]).T

    # transform
    pgLift = liftPts(gps, scps) # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)
    ygt = np.dot(pgLift, cy.T)

    return np.vstack([xgt, ygt]).T

def image_tps_transform(image: np.ndarray, scps, tcps, keep_filled= True):
    """ Given a image with shape (C, H, W), perform TPS transform on this image
    @ Args:
        image: np.ndarray with shape (C, H, W)
        scps: np.ndarray with shape (N, 2) specifying interest points in (x,y) coordinates
        tcps: np.ndarray with shape (N, 2) telling target points
        keep_filled: if True, will add 4 control points to keep all pixels are stayed in the frame
    @ Returns:
        result: image after TPS transformation
    """
    C, H, W = image.shape
    if keep_filled:
        additional_cps = np.array([
            [0., 0.],
            [0., H-1],
            [W-1, 0.],
            [W-1, H-1],
        ])
        scps = np.vstack([scps, additional_cps])
        tcps = np.vstack([tcps, additional_cps])
    all_tcps = get_all_tcps(scps, tcps, image.shape[1:])

    all_tcps = np.rint(all_tcps)
    all_tcps[:,0] = np.clip(all_tcps[:,0], 0, W-1)
    all_tcps[:,1] = np.clip(all_tcps[:,1], 0, H-1)
    all_tcps = all_tcps.astype(np.int32)

    # result = np.zeros_like(image)
    # result = image.copy()
    # result[:, all_tcps[:,1], all_tcps[:,0]] = image.reshape(C, -1)
    result = image[:, all_tcps[:,1], all_tcps[:,0]].reshape(C, H, W)
    return result

