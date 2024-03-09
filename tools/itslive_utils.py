import json
import logging
import time

import numpy as np
import pandas as pd
import pyproj
import s3fs as s3
import shapefile
import xarray as xr
from shapely import geometry

from tools.utils import xy2ll

logging.basicConfig(level=logging.ERROR)


def flowline_sample(flowline_file,
                    epsg='3031',
                    Npts=30,
                    variables=['v', 'v_error'],
                    resample={"mid_date": '1d'},
                    concat_dim='x'):
    """
    Sample ITS_LIVE ice velocity data along an arbitrary line (flowline)

    Parameters
    ---
    flowline_file : str
        Path to .shp/.shx file that is a multi-line or line shapefile
    epsg : str (default = '3031')
        EPSG the shapefile is in
    variables : list of str (default = ['v', 'v_error'])
        List of variables to sample from a xarray dataset
    resample : dict (default = {'mid_date':'1d'}
    concat_dim : str (default = 'x')
        Concatenate the results over a dimension of the xarray dataset.
    Returns
    ---
    dss : list of xarray datasets
    """
    DT = DATACUBETOOLS()
    flowline = shapefile.Reader(flowline_file)
    shapes = flowline.shapes()

    dss = []
    for fl in shapes:
        pts = np.array(fl.points)
        n = pts.shape[0]
        # sample roughly evenly spaced points
        ix = np.linspace(0, n - 1, Npts, dtype=int)
        ix[ix >= n] = n - 1
        ix = np.unique(ix)
        pts = pts[ix]

        tmp = []
        for i, (xi, yi) in enumerate(pts):
            # _, ds, _ = DT.get_timeseries_at_point((xi, yi), epsg, variables=variables)
            loni, lati = xy2ll(xi, yi)
            _, ds, _ = DT.get_timeseries_at_point((loni, lati), "4326", variables=variables)
            # have to sort because the ITS_LIVE data isn't natively sorted by date
            # then we have to resample by some frequency to do easier analyses
            ds = ds.sortby(variables='mid_date').resample(resample, skipna=False).mean()
            tmp.append(ds)
        ds = xr.combine_nested(tmp, concat_dim=concat_dim)
        dss.append(ds)
    if len(dss) == 1:
        return ds
    else:
        return dss


########################################################################################################################
# Below here is some code to calculate stress and strain from surface velocity rasters. The overwhelming majority of
# this code is copied directly from Bryan Vriel's iceutils (https://github.com/bryanvriel/iceutils/blob/master/iceutils).
# Only reproduced here with minimal changes for my own use.
########################################################################################################################

def _make_odd(w):
    """
    Convenience function to ensure a number is odd.
    """
    if w % 2 == 0:
        w += 1
    return w


# Lambda for getting order-dependent polynomial exponents
_compute_exps = lambda order: [(k - n, n) for k in range(order + 1) for n in range(k + 1)]


def gradient(z, spacing=1.0, axis=None, remask=True, method='sgolay',
             inpaint=False, **kwargs):
    """
    Calls either Numpy or Savitzky-Golay gradient computation routines.
    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple, spacing corresponds
        to axes directions specified by axis. Default: 1.0.
    axis: None or int or tuple of ints, optional
        Axis or axes to compute gradient. If None, derivative computed along all
        dimensions. Default: 0.
    remask: bool, optional
        Apply NaN mask on gradients. Default: True.
    method: str, optional
        Method specifier in ('numpy', 'sgolay', 'robust'). Default: 'numpy'.
    inpaint: bool, optional
        Inpaint image prior to gradient computation (recommended for
        'sgolay' method). Default: False.
    **kwargs:
        Extra keyword arguments to pass to specific gradient computation.
    Returns
    -------
    s: ndarray or list of ndarray
        Set of ndarrays (or single ndarry for only one axis) with same shape as z
        corresponding to the derivatives of z with respect to each axis.
    """
    # Mask mask of NaNs
    nan_mask = np.isnan(z)
    have_nan = np.any(nan_mask)

    # For sgolay method, we need to inpaint if NaNs detected
    if inpaint and have_nan:
        print('in painting nans')
        z_inp = _inpaint(z, mask=nan_mask)
    else:
        z_inp = z

    # Compute gradient with numpy
    if method == 'numpy':
        if isinstance(spacing, (tuple, list)):
            s = np.gradient(z_inp, spacing[0], spacing[1], axis=(0, 1), edge_order=2)
        else:
            s = np.gradient(z_inp, spacing, axis=axis, edge_order=2)

    # With Savtizky-Golay
    elif method == 'sgolay':
        s = sgolay_gradient(z_inp, spacing=spacing, axis=axis, **kwargs)

    # With robust polynomial
    elif method in ('robust_l2', 'robust_lp'):
        zs, z_dy, z_dx = robust_gradient(z_inp, spacing=spacing, lsq_method=method, **kwargs)
        s = (z_dy, z_dx)
        if axis is not None and isinstance(axis, int):
            s = s[axis]

    else:
        raise ValueError('Unsupported gradient method.')

    # Re-apply mask
    if remask and have_nan:
        if isinstance(s, (tuple, list)):
            for arr in s:
                arr[nan_mask] = np.nan
        else:
            s[nan_mask] = np.nan

    return s


def sgolay_gradient(z, spacing=1.0, axis=None, window_size=5, order=2):
    """
    Wrapper around Savitzky-Golay code to compute window size in pixels and call _sgolay2d
    with correct arguments.
    Parameters
    ----------
    z: array_like
        2-dimensional array containing samples of a scalar function.
    spacing: float or tuple of floats, optional
        Spacing between f values along specified axes. If tuple provided, spacing is
        specified as (dy, dx) and derivative computed along both dimensions.
        Default is unitary spacing.
    axis: None or int, optional
        Axis along which to compute gradients. If None, gradient computed
        along both dimensions. Default: None.
    window_size: scalar or tuple of scalars, optional
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x). Default: 3.
    order: int, optional
        Polynomial order. Default: 4.
    Returns
    -------
    gradient: a
        Array or tuple of array corresponding to gradients. If both axes directions
        are specified, returns (dz/dy, dz/dx).
    """
    # Compute derivatives in both directions
    if axis is None or isinstance(spacing, (tuple, list)):

        # Compute window sizes
        wy, wx = compute_windows(window_size, spacing)

        # Unpack spacing
        dy, dx = spacing

        # Call Savitzky-Golay twice in order to use different window sizes
        sy = _sgolay2d(z, window_size, order=order, derivative='col')
        sx = _sgolay2d(z, window_size, order=order, derivative='row')

        # Scale by spacing and return
        return sy / dy, sx / dx

    # Or derivative in a single direction
    else:

        assert axis is not None, 'Must specify axis direction.'

        # Compute window size
        w = int(np.ceil(abs(window_size / spacing)))
        if w % 2 == 0:
            w += 1

        # Call Savitzky-Golay
        if axis == 0:
            s = _sgolay2d(z, w, order=order, derivative='col')
        elif axis == 1:
            s = _sgolay2d(z, w, order=order, derivative='row')
        else:
            raise ValueError('Axis must be 0 or 1.')

        # Scale by spacing and return
        return s / spacing


def _sgolay2d(z, window_size=5, order=2, derivative='both'):
    """
    Max Filter, January 2021.
    Original lower-level code from Scipy cookbook, with modifications to
    padding.
    """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size ** 2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2, )

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    Z = np.pad(z, half_size, mode='reflect')

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        Zf = fftconvolve(Z, m, mode='valid')
        return Zf
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        return Zc
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        Zc = fftconvolve(Z, -c, mode='valid')
        Zr = fftconvolve(Z, -r, mode='valid')
        return Zr, Zc


def compute_windows(window_size, spacing):
    """
    Convenience function to compute window sizes in pixels given spacing of
    pixels in physical coordinates.
    Parameters
    ----------
    spacing: scalar or tuple of scalars
        Spacing between pixels along axis. If tuple, each element specifies
        spacing along different axes.
    window_size: scalar or tuple of scalars
        Window size in units of specified spacing. If tuple provided, window size is
        specified as (win_y, win_x).
    Returns
    -------
    w: tuple of scalars
        Odd-number window sizes in both axes directions in number of pixels.
    """
    # Unpack spacing
    if isinstance(spacing, (tuple, list)):
        assert len(spacing) == 2, 'Spacing must be 2-element tuple.'
        dy, dx = spacing
    else:
        dy = dx = spacing

    # Compute window sizes
    if isinstance(window_size, (tuple, list)):
        assert len(window_size) == 2, 'Window size must be 2-element tuple.'
        wy, wx = window_size
    else:
        wy = wx = window_size

    # Array of windows
    if isinstance(wy, np.ndarray):
        wy = np.ceil(np.abs(wy / dy)).astype(int)
        wx = np.ceil(np.abs(wx / dx)).astype(int)
        wy[(wy % 2) == 0] += 1
        wx[(wx % 2) == 0] += 1

    # Or scalar windows
    else:
        wy, wx = int(np.ceil(abs(wy / dy))), int(np.ceil(abs(wx / dx)))
        wy = _make_odd(wy)
        wx = _make_odd(wx)

    return wy, wx


class Raster:
    def __int__(self):
        pass


def _unique_rows(scenes):
    '''Unique rows utility similar to matlab.'''
    uscenes = np.unique(scenes.view([('', scenes.dtype)] * scenes.shape[1])).view(scenes.dtype).reshape(-1,
                                                                                                        scenes.shape[1])
    return uscenes


def _inpaint_spring(ain, mask):
    '''Returns the inpainted matrix using the spring metaphor.
       All NaN values in the matrix are filled in.

       Based on the original inpaintnans package by John D'Errico.
       http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaintnans'''
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    dims = ain.shape
    bout = ain.copy()
    nnn = dims[0]
    mmm = dims[1]
    nnmm = nnn * mmm

    [iii, jjj] = np.where(mask == False)
    [iin, jjn] = np.where(mask == True)
    nnan = len(iin)  # Number of nan.

    if nnan == 0:
        return bout

    hv_springs = np.zeros((4 * nnan, 2), dtype=int)
    cnt = 0
    for kkk in range(nnan):
        ypos = iin[kkk]
        xpos = jjn[kkk]
        indc = ypos * mmm + xpos
        if (ypos > 0):
            hv_springs[cnt, :] = [indc - mmm, indc]  # Top
            cnt = cnt + 1

        if (ypos < (nnn - 1)):
            hv_springs[cnt, :] = [indc, indc + mmm]  # Bottom
            cnt = cnt + 1

        if (xpos > 0):
            hv_springs[cnt, :] = [indc - 1, indc]  # Left
            cnt = cnt + 1

        if (xpos < (mmm - 1)):
            hv_springs[cnt, :] = [indc, indc + 1]  # Right
            cnt = cnt + 1

    hv_springs = hv_springs[0:cnt, :]

    tempb = _unique_rows(hv_springs)
    cnt = tempb.shape[0]

    alarge = sp.csc_matrix((np.ones(cnt), (np.arange(cnt), tempb[:, 0])),
                           shape=(cnt, nnmm))
    alarge = alarge + sp.csc_matrix((-np.ones(cnt), (np.arange(cnt)
                                                     , tempb[:, 1])), shape=(cnt, nnmm))

    indk = iii * mmm + jjj
    indu = iin * mmm + jjn
    dkk = -ain[iii, jjj]
    del iii
    del jjj

    aknown = sp.csc_matrix(alarge[:, indk])
    rhs = sp.csc_matrix.dot(aknown, dkk)
    del aknown
    del dkk
    anan = sp.csc_matrix(alarge[:, indu])
    dku = sla.lsqr(anan, rhs)
    bout[iin, jjn] = dku[0]
    return bout


def _inpaint(raster, mask=None, method='spring', r=3.0):
    """
    Inpaint a raster at NaN values or with an input mask.
    Parameters
    ----------
    raster: Raster or ndarray
        Input raster or array object to inpaint.
    mask: None or ndarry, optional
        Mask with same shape as raster specifying pixels to inpaint. If None,
        mask computed from NaN values. Default: None.
    method: str, optional
        Inpainting method from ('telea', 'biharmonic'). Default: 'telea'.
    r: scalar, optional
        Radius in pixels of neighborhood for OpenCV inpainting. Default: 3.0.
    Returns
    -------
    out_raster: Raster
        Output raster object.
    """
    if isinstance(raster, Raster):
        rdata = raster.data
    else:
        rdata = raster

    # Create mask
    if mask is None:
        mask = np.isnan(rdata)
    else:
        assert mask.shape == rdata.shape, 'Mask and raster shape mismatch.'

    # Check suitability of inpainting method with available packages
    if method == 'telea' and cv is None:
        warnings.warn('OpenCV package cv2 not found; falling back to spring inpainting.')
        method = 'spring'

    # Call inpainting
    if method == 'spring':
        inpainted = _inpaint_spring(rdata, mask)
    elif method == 'telea':
        umask = mask.astype(np.uint8)
        inpainted = cv.inpaint(rdata, umask, r, cv.INPAINT_TELEA)
    elif method == 'biharmonic':
        inpainted = inpaint_biharmonic(rdata, mask, multichannel=False)
    else:
        raise ValueError('Unsupported inpainting method.')

    # Return new raster or array
    if isinstance(raster, Raster):
        return Raster(data=inpainted, hdr=raster.hdr)
    else:
        return inpainted


def compute_strains(vx, vy, dx=240, dy=-240, window_size=5, grad_method='numpy', rotate=True, inpaint=False, **kwargs):
    """
    Calculate the surface strain based on ice surface velocities.

    Based upon code develped by J. Millstein (github.com/jdmillstein/strain_rates)

    Parameters
    ---
    vx : (Ny,Nx) array
        x-component of velocity
    vy : (Ny,Nx) array
        y-component of velocity
    dx : float
        Spatial step size in x-direction
    dy : float
        spatial step size in y-direction
    window_size : int

    grad_method : str
        Method to use for calculating velocity gradients
    rotate : bool
        Whether to rotate or not
    kwargs : dict?

    Returns
    ---
    strain_dict : dictionary

    D : np.ndarray

    """

    Ny, Nx = vx.shape

    # Compute velocity gradients
    L12, L11 = gradient(vx, spacing=(dy, dx), method=grad_method, inpaint=inpaint, window_size=window_size, **kwargs)
    L22, L21 = gradient(vy, spacing=(dy, dx), method=grad_method, inpaint=inpaint, window_size=window_size, **kwargs)

    # Compute components of strain-rate tensor
    D = np.empty((2, 2, vx.size))
    D[0, 0, :] = 0.5 * (L11 + L11).ravel()
    D[0, 1, :] = 0.5 * (L12 + L21).ravel()
    D[1, 0, :] = 0.5 * (L21 + L12).ravel()
    D[1, 1, :] = 0.5 * (L22 + L22).ravel()

    # Compute pixel-dependent rotation tensor if requested
    if rotate:
        R = np.empty((2, 2, vx.size))
        theta = np.arctan2(vy, vx).ravel()
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)

        # Apply rotation tensor
        D = np.einsum('ijm,kjm->ikm', D, R)
        D = np.einsum('ijm,jkm->ikm', R, D)

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]

    # Normal strain rates
    exx = D11.reshape(Ny, Nx)
    eyy = D22.reshape(Ny, Nx)

    # Shear- same result as: e_xy_max = np.sqrt(0.25 * (e_x - e_y)**2 + e_xy**2)
    trace = D11 + D22
    det = D11 * D22 - D12 * D21
    shear = np.sqrt(0.25 * trace ** 2 - det).reshape(Ny, Nx)

    # Compute scalar quantities from stress tensors
    dilatation = (L11 + L22).reshape(Ny, Nx)
    effective_strain = np.sqrt(L11 ** 2 + L22 ** 2 + 0.25 * (L12 + L21) ** 2 + L11 * L22).reshape(Ny, Nx)

    # Store strain components in dictionary
    strain_dict = {'e_xx': exx,
                   'e_yy': eyy,
                   'e_xy': shear,
                   'dilatation': dilatation,
                   'effective': effective_strain}

    # Return strain and stress dictionaries
    return strain_dict, D


def compute_stress_strain(vx, vy, dx=240, dy=-240, window_size=31, grad_method='numpy', rotate=True, inpaint=True,
                          AGlen=None, n=3, h=None, b=None, **kwargs):
    Ny, Nx = vx.shape
    strain_dict, D = compute_strains(vx, vy, dx, dy, window_size, grad_method, rotate, inpaint)

    if AGlen is None:
        if 'T' in kwargs:
            T = kwargs['T'] + 273.15
        else:
            T = -23. + 273.15
        a0 = 5.3e-15 * 365.25 * 24. * 60. * 60.  # a-1 kPa-3
        fa = 1.2478766e-39 * np.exp(0.32769 * T) + 1.9463011e-10 * np.exp(0.07205 * T)
        AGlen = a0 * fa * 1e-9

    # Cache elements of strain-rate tensor for easier viewing
    D11 = D[0, 0, :]
    D12 = D[0, 1, :]
    D21 = D[1, 0, :]
    D22 = D[1, 1, :]

    scale_factor = 0.5 * AGlen ** (-1. / n)
    eta = scale_factor * strain_dict['effective'] ** ((1.0 - n) / n)
    txx = 2.0 * eta * D11.reshape(Ny, Nx)
    tyy = 2.0 * eta * D22.reshape(Ny, Nx)
    txy = 2.0 * eta * D12.reshape(Ny, Nx)
    tyx = 2.0 * eta * D21.reshape(Ny, Nx)

    stress_dict = {'eta': eta,
                   't_xx': txx,
                   't_yy': tyy,
                   't_xy': 0.5 * (txy + tyx)
                   }

    if h is not None:
        if b is not None:
            s = h + b
            s_x = gradient(s, dx, axis=1, method=grad_method, inpaint=inpaint, remask=False)
            s_y = gradient(s, dy, axis=0, method=grad_method, inpaint=inpaint, remask=False)
        else:
            s_x = gradient(h, dx, axis=1, method=grad_method, inpaint=inpaint, remask=False)
            s_y = gradient(h, dy, axis=0, method=grad_method, inpaint=inpaint, remask=False)

        # Membrane stresses
        tmxx = gradient(h * (2 * txx + tyy), dx, axis=1, method=grad_method,
                        inpaint=inpaint)
        tmxy = gradient(h * 0.5 * (txy + tyx), dy, axis=0, method=grad_method,
                        inpaint=inpaint)
        tmyy = gradient(h * (2 * tyy + txx), dy, axis=0, method=grad_method,
                        inpaint=inpaint)
        tmyx = gradient(h * 0.5 * (txy + tyx), dx, axis=1, method=grad_method,
                        inpaint=inpaint)

        # Driving stresses
        tdx = -1.0 * 917. * 9.81 * h * s_x
        tdy = -1.0 * 917. * 9.81 * h * s_y

        # Optional rotation of driving stress
        if rotate:
            # Compute unit vectors
            vmag = np.sqrt(vx ** 2 + vy ** 2)
            uhat = vx / vmag
            vhat = vy / vmag

            # Rotate to along-flow, across-flow
            tdx2 = tdx * uhat + tdy * vhat
            tdy2 = tdx * (-vhat) + tdy * uhat
            tdx, tdy = tdx2, tdy2

        # Pack extra stress components
        stress_dict['tmxx'] = tmxx
        stress_dict['tmxy'] = tmxy
        stress_dict['tmyy'] = tmyy
        stress_dict['tmyx'] = tmyx
        stress_dict['tdx'] = tdx
        stress_dict['tdy'] = tdy

    return strain_dict, stress_dict, D


########################################################################################################################
# Below here is some code for downloading and using ITS_LIVE velocity pair data points. The overwhelming majority of
# this code is copied directly from JPL's ITS_LIVE code (https://github.com/nasa-jpl/its_live_production). Only
# reproduced here with minimal changes for my own use
########################################################################################################################

# class to throw time series lookup errors
class timeseriesException(Exception):
    pass


class DATACUBETOOLS:
    """
    class to encapsulate discovery and interaction with ITS_LIVE (its-live.jpl.nasa.gov) datacubes on AWS s3
    """

    VELOCITY_DATA_ATTRIBUTION = """ \nITS_LIVE velocity data
    (<a href="https://its-live.jpl.nasa.gov">ITS_LIVE</a>) with funding provided by NASA MEaSUREs.\n
    """

    def __init__(self, use_catalog="all"):
        """
        tools for accessing ITS_LIVE glacier velocity datacubes in S3
        __init__ reads in the geojson catalog of datacubes and creates list .open_cubes
        """
        # the URL for the current datacube catalog GeoJSON file - set up as dictionary to allow other catalogs for testing
        self.catalog = {
            "all": "s3://its-live-data/datacubes/catalog_v02.json",
        }

        # S3fs used to access cubes in python
        self._s3fs = s3.S3FileSystem(anon=True)
        # keep track of open cubes so that we don't re-read xarray metadata and dimension vectors
        self.open_cubes = {}
        self._current_catalog = use_catalog
        with self._s3fs.open(self.catalog[use_catalog], "r") as incubejson:
            self._json_all = json.load(incubejson)
        self.json_catalog = self._json_all

    def find_datacube_catalog_entry_for_point(self, point_xy, point_epsg_str):
        """
        find catalog feature that contains the point_xy [x,y] in projection point_epsg_str (e.g. '3413')
        returns the catalog feature and the point_tilexy original point coordinates reprojected into the datacube's native projection
        (cubefeature, point_tilexy)
        """
        if point_epsg_str != "4326":
            # point not in lon,lat, set up transformation and convert it to lon,lat (epsg:4326)
            # because the features in the catalog GeoJSON are polygons in 4326
            inPROJtoLL = pyproj.Transformer.from_proj(
                f"epsg:{point_epsg_str}", "epsg:4326", always_xy=True
            )
            pointll = inPROJtoLL.transform(*point_xy)
        else:
            # point already lon,lat
            pointll = point_xy

        # create Shapely point object for inclusion test
        point = geometry.Point(*pointll)  # point.coords.xy

        # find datacube outline that contains this point in geojson index file
        cubefeature = None

        for f in self.json_catalog["features"]:
            polygeom = geometry.shape(f["geometry"])
            if polygeom.contains(point):
                cubefeature = f
                break

        if cubefeature:
            # find point x and y in cube native epsg if not already in that projection
            if point_epsg_str == str(cubefeature["properties"]["epsg"]):
                point_cubexy = point_xy
            else:
                inPROJtoTilePROJ = pyproj.Transformer.from_proj(
                    f"epsg:{point_epsg_str}",
                    f"EPSG:{cubefeature['properties']['epsg']}",
                    always_xy=True,
                )
                point_cubexy = inPROJtoTilePROJ.transform(*point_xy)

            # print(
            #     f"original xy {point_xy} {point_epsg_str} maps to datacube {point_cubexy} "
            #     f"EPSG:{cubefeature['properties']['epsg']}"
            # )

            # now test if point is in xy box for cube (should be most of the time; could fail
            # because of boundary curvature 4326 box defined by lon,lat corners but point needs to be in box defined in cube's projection)
            #
            point_cubexy_shapely = geometry.Point(*point_cubexy)
            polygeomxy = geometry.shape(cubefeature["properties"]["geometry_epsg"])
            if not polygeomxy.contains(point_cubexy_shapely):
                # first find cube proj bounding box
                dcbbox = np.array(
                    cubefeature["properties"]["geometry_epsg"]["coordinates"][0]
                )
                minx = np.min(dcbbox[:, 0])
                maxx = np.max(dcbbox[:, 0])
                miny = np.min(dcbbox[:, 1])
                maxy = np.max(dcbbox[:, 1])

                # point is in lat lon box, but not in cube-projection's box
                # try once more to find proper cube by using a new point in cube projection moved 10 km farther from closest
                # boundary in cube projection; use new point's lat lon to search for new cube - test if old point is in that
                # new cube's projection box, otherwise ...
                # this next section tries one more time to find new feature after offsetting point farther outside box of
                # first cube, in cube projection, to deal with curvature of lat lon box edges in different projections
                #
                # point in ll box but not cube_projection box, move point in cube projection
                # 10 km farther outside box, find new ll value for point, find new feature it is in,
                # and check again if original point falls in this new cube's
                # move coordinate of point outside this box farther out by 10 km

                newpoint_cubexy = list(point_cubexy)
                if point_cubexy[1] < miny:
                    newpoint_cubexy[1] -= 10000.0
                elif point_cubexy[1] > maxy:
                    newpoint_cubexy[1] += 10000.0
                elif point_cubexy[0] < minx:
                    newpoint_cubexy[0] -= 10000.0
                elif point_cubexy[0] > maxx:
                    newpoint_cubexy[0] += 10000.0
                else:
                    # no change has been made to newpoint_cubexy because
                    # user has chosen a point exactly on the boundary, move it 1 m into the box...
                    logging.info(
                        "user has chosen a point exactly on the boundary, move it 1 m into the box..."
                    )
                    if point_cubexy[1] == miny:
                        newpoint_cubexy[1] += 1.0
                    elif point_cubexy[1] == maxy:
                        newpoint_cubexy[1] -= 1.0
                    elif point_cubexy[0] == minx:
                        newpoint_cubexy[0] += 1.0
                    elif point_cubexy[0] == maxx:
                        newpoint_cubexy[0] -= 1.0

                # now reproject this point to lat lon and look for new feature

                cubePROJtoLL = pyproj.Transformer.from_proj(
                    f'{cubefeature["properties"]["data_epsg"]}',
                    "epsg:4326",
                    always_xy=True,
                )
                newpointll = cubePROJtoLL.transform(*newpoint_cubexy)

                # create Shapely point object for inclusion test
                newpoint = geometry.Point(*newpointll)

                # find datacube outline that contains this point in geojson index file
                newcubefeature = None

                for f in self.json_catalog["features"]:
                    polygeom = geometry.shape(f["geometry"])
                    if polygeom.contains(newpoint):
                        newcubefeature = f
                        break

                if newcubefeature:
                    # if new feature found, see if original (not offset) point is in this new cube's cube-projection bounding box
                    # find point x and y in cube native epsg if not already in that projection
                    if (
                            cubefeature["properties"]["data_epsg"]
                            == newcubefeature["properties"]["data_epsg"]
                    ):
                        point_cubexy = newpoint_cubexy
                    else:
                        # project original point in this new cube's projection
                        inPROJtoTilePROJ = pyproj.Transformer.from_proj(
                            f"epsg:{point_epsg_str}",
                            newcubefeature["properties"]["data_epsg"],
                            always_xy=True,
                        )
                        point_cubexy = inPROJtoTilePROJ.transform(*point_xy)

                    logging.info(
                        f"try 2 original xy {point_xy} {point_epsg_str} with offset maps to new datacube {point_cubexy} "
                        f" {newcubefeature['properties']['data_epsg']}"
                    )

                    # now test if point is in xy box for cube (should be most of the time;
                    #
                    point_cubexy_shapely = geometry.Point(*point_cubexy)
                    polygeomxy = geometry.shape(
                        newcubefeature["properties"]["geometry_epsg"]
                    )
                    if not polygeomxy.contains(point_cubexy_shapely):
                        # point is in lat lon box, but not in cube-projection's box
                        # try once more to find proper cube by using a new point in cube projection moved 10 km farther from closest
                        # boundary in cube projection; use new point's lat lon to search for new cube - test if old point is in that
                        # new cube's projection box, otherwise fail...

                        raise timeseriesException(
                            f"point is in lat,lon box but not {cubefeature['properties']['data_epsg']} box!! even after offset"
                        )
                    else:
                        return (newcubefeature, point_cubexy)

            else:
                return (cubefeature, point_cubexy)

        else:
            print(f"No data for point (lon,lat) {pointll}")
            return (None, None)

    def get_timeseries_at_point(self, point_xy, point_epsg_str, variables=["v"]):
        """pulls time series for a point (closest ITS_LIVE point to given location):
        - calls find_datacube to determine which S3-based datacube the point is in,
        - opens that xarray datacube - which is also added to the open_cubes list, so that it won't need to be reopened (which can take O(5 sec) ),
        - extracts time series at closest grid cell to the original point
            (time_series.x and time_series.y contain x and y coordinates of ITS_LIVE grid cell in datacube projection)

        returns(
            - xarray of open full cube (not loaded locally, but coordinate vectors and attributes for full cube are),
            - time_series (as xarray dataset with all requested variables, that is loaded locally),
            - original point xy in datacube's projection
            )

        NOTE - returns an xarray Dataset (not just a single xarray DataArray) - time_series.v or time_series['v'] is speed
        """

        start = time.time()

        cube_feature, point_cubexy = self.find_datacube_catalog_entry_for_point(
            point_xy, point_epsg_str
        )

        if cube_feature is None:
            return (None, None, None)

        # for zarr store modify URL for use in boto open - change http: to s3: and lose s3.amazonaws.com
        incubeurl = (
            cube_feature["properties"]["zarr_url"]
            .replace("http:", "s3:")
            .replace(".s3.amazonaws.com", "")
        )

        # if we have already opened this cube, don't open it again
        if len(self.open_cubes) > 0 and incubeurl in self.open_cubes.keys():
            ins3xr = self.open_cubes[incubeurl]
        else:
            ins3xr = xr.open_dataset(
                incubeurl, engine="zarr", storage_options={"anon": True}
            )
            self.open_cubes[incubeurl] = ins3xr

        # find time series at the closest grid cell
        # NOTE - returns an xarray Dataset - pt_dataset.v is speed...
        pt_datset = ins3xr[variables].sel(
            x=point_cubexy[0], y=point_cubexy[1], method="nearest"
        )

        logging.info(
            f"xarray open - elapsed time: {(time.time() - start):10.2f}", flush=True
        )

        # pull data to local machine
        pt_datset.load()

        # print(
        #     f"time series loaded {[f'{x}: {pt_datset[x].shape[0]}' for x in variables]} points - elapsed time: {(time.time()-start):10.2f}",
        #     flush=True,
        # )
        # end for zarr store

        return (ins3xr, pt_datset, point_cubexy)

    def set_mapping_for_small_cube_from_larger_one(self, smallcube, largecube):
        """when a subset is pulled from an ITS_LIVE datacube, a new geotransform needs to be
        figured out from the smallcube's x and y coordinates and stored in the GeoTransform attribute
        of the mapping variable (which also needs to be copied from the original cube)
        """
        largecube_gt = [float(x) for x in largecube.mapping.GeoTransform.split(" ")]
        smallcube_gt = largecube_gt  # need to change corners still
        # find UL corner of UL pixel (x and y are pixel center coordinates)
        smallcube_gt[0] = smallcube.x.min().item() - (
                smallcube_gt[1] / 2.0
        )  # set new ul x value
        smallcube_gt[3] = smallcube.y.max().item() - (
                smallcube_gt[5] / 2.0
        )  # set new ul y value
        smallcube[
            "mapping"
        ] = largecube.mapping  # still need to add new GeoTransform as string
        smallcube.mapping["GeoTransform"] = " ".join([str(x) for x in smallcube_gt])
        return

    def get_subcube_around_point(
            self, point_xy, point_epsg_str, half_distance=5000.0, variables=["v"]
    ):
        """pulls subset of cube within half_distance of point (unless edge of cube is included) containing specified variables:
        - calls find_datacube to determine which S3-based datacube the point is in,
        - opens that xarray datacube - which is also added to the open_cubes list, so that it won't need to be reopened (which can take O(5 sec) ),
        - extracts smaller cube containing full time series of specified variables

        returns(
            - xarray of open full cube (not loaded locally, but coordinate vectors and attributes for full cube are),
            - smaller cube as xarray,
            - original point xy in datacube's projection
            )
        """

        start = time.time()

        cube_feature, point_cubexy = self.find_datacube_catalog_entry_for_point(
            point_xy, point_epsg_str
        )

        # for zarr store modify URL for use in boto open - change http: to s3: and lose s3.amazonaws.com
        incubeurl = (
            cube_feature["properties"]["zarr_url"]
            .replace("http:", "s3:")
            .replace(".s3.amazonaws.com", "")
        )

        # if we have already opened this cube, don't open it again
        if len(self.open_cubes) > 0 and incubeurl in self.open_cubes.keys():
            ins3xr = self.open_cubes[incubeurl]
        else:
            ins3xr = xr.open_dataset(
                incubeurl, engine="zarr", storage_options={"anon": True}
            )
            self.open_cubes[incubeurl] = ins3xr

        pt_tx, pt_ty = point_cubexy
        lx = ins3xr.coords["x"]
        ly = ins3xr.coords["y"]

        start = time.time()
        small_ins3xr = (
            ins3xr[variables]
            .loc[
                dict(
                    x=lx[(lx > pt_tx - half_distance) & (lx < pt_tx + half_distance)],
                    y=ly[(ly > pt_ty - half_distance) & (ly < pt_ty + half_distance)],
                )
            ]
            .load()
        )
        print(f"subset and load at {time.time() - start:6.2f} seconds", flush=True)

        # now fix the CF compliant geolocation/mapping of the smaller cube
        self.set_mapping_for_small_cube_from_larger_one(small_ins3xr, ins3xr)

        return (ins3xr, small_ins3xr, point_cubexy)

    def get_subcube_for_bounding_box(self, bbox, bbox_epsg_str, variables=["v"]):
        """pulls subset of cube within bbox (unless edge of cube is included) containing specified variables:
        - calls find_datacube to determine which S3-based datacube the bbox central point is in,
        - opens that xarray datacube - which is also added to the open_cubes list, so that it won't need to be reopened (which can take O(5 sec) ),
        - extracts smaller cube containing full time series of specified variables

        bbox = [ minx, miny, maxx, maxy ] in bbox_epsg_str meters
        bbox_epsg_str = '3413', '32607', '3031', ... (EPSG:xxxx) projection identifier
        variables = [ 'v', 'vx', 'vy', ...] variables in datacube - note 'mapping' is returned by default, with updated geotransform attribute for the new subcube size

        returns(
            - xarray of open full cube (not loaded locally, but coordinate vectors and attributes for full cube are),
            - smaller cube as xarray (loaded to memory),
            - original bbox central point xy in datacube's projection
            )
        """

        start = time.time()

        #
        # derived from point/distance (get_subcube_around_point) so first iteration uses central point to look up datacube to open
        # subcube will still be clipped at datacube edge if bbox extends to other datacubes - in future maybe return subcubes from each?
        #
        # bbox is probably best expressed in datacube epsg - will fail if different...  in future, deal with this some other way.
        #

        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = bbox
        bbox_centrer_point_xy = [
            (bbox_minx + bbox_maxx) / 2.0,
            (bbox_miny + bbox_maxy) / 2.0,
        ]
        (
            cube_feature,
            bbox_centrer_point_cubexy,
        ) = self.find_datacube_catalog_entry_for_point(
            bbox_centrer_point_xy, bbox_epsg_str
        )

        if cube_feature["properties"]["data_epsg"].split(":")[-1] != bbox_epsg_str:
            print(
                f'bbox is in epsg:{bbox_epsg_str}, should be in datacube {cube_feature["properties"]["data_epsg"]}'
            )
            return None

        # for zarr store modify URL for use in boto open - change http: to s3: and lose s3.amazonaws.com
        incubeurl = (
            cube_feature["properties"]["zarr_url"]
            .replace("http:", "s3:")
            .replace(".s3.amazonaws.com", "")
        )

        # if we have already opened this cube, don't open it again
        if len(self.open_cubes) > 0 and incubeurl in self.open_cubes.keys():
            ins3xr = self.open_cubes[incubeurl]
        else:
            # open zarr format xarray datacube on AWS S3
            ins3xr = xr.open_dataset(
                incubeurl, engine="zarr", storage_options={"anon": True}
            )
            self.open_cubes[incubeurl] = ins3xr

        lx = ins3xr.coords["x"]
        ly = ins3xr.coords["y"]

        start = time.time()
        small_ins3xr = (
            ins3xr[variables]
            .loc[
                dict(
                    x=lx[(lx >= bbox_minx) & (lx <= bbox_maxx)],
                    y=ly[(ly >= bbox_miny) & (ly <= bbox_maxy)],
                )
            ]
            .load()
        )
        print(f"subset and load at {time.time() - start:6.2f} seconds", flush=True)

        # now fix the CF compliant geolocation/mapping of the smaller cube
        self.set_mapping_for_small_cube_from_larger_one(small_ins3xr, ins3xr)

        return (ins3xr, small_ins3xr, bbox_centrer_point_cubexy)


def running_mean(mid_dates, variable, minpts, tFreq):
    """
    Taken directly from datacube_tools.py @
    https://github.com/nasa-jpl/its_live/blob/main/notebooks/


    mid_dates: center dates of `variable` data [datetime64]
    variable: data to be average
    minpts: minimum number of points needed for a valid value, else filled with nan
    tFreq: the spacing between centered averages in Days, default window size = tFreq*2
    """
    tsmin = pd.Timestamp(np.min(mid_dates))
    tsmax = pd.Timestamp(np.max(mid_dates))
    ts = pd.date_range(start=tsmin, end=tsmax, freq=f"{tFreq}D")
    ts = pd.to_datetime(ts).values
    idx0 = ~np.isnan(variable)
    runmean = np.empty([len(ts) - 1, 1])
    runmean[:] = np.nan
    tsmean = ts[0:-1]

    t_np = mid_dates.astype(np.int64)

    for i in range(len(ts) - 1):
        idx = (
                (mid_dates >= (ts[i] - np.timedelta64(int(tFreq / 2), "D")))
                & (mid_dates < (ts[i + 1] + np.timedelta64(int(tFreq / 2), "D")))
                & idx0
        )
        if sum(idx) >= minpts:
            runmean[i] = np.mean(variable[idx])
            tsmean[i] = np.mean(t_np[idx])

    tsmean = pd.to_datetime(tsmean).values
    return (runmean, tsmean)
