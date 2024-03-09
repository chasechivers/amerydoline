import numpy as np
import rasterio as rio
import xarray as xr

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

def inpaint_nans(array):
    """
    Mask out nan values in array

    Parameters
    ---
    array : np.ndarray

    Returns
    ---
    filled : np.array
    
    """
    valid_mask = ~np.isnan(array)
    coords = np.array(np.nonzero(valid_mask)).T
    values = array[valid_mask]
    it = LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(array.shape))).reshape(array.shape)
    return filled


def dem_and_xy(dem_file):
    """

    Parameters
    ---
    dem_file:

    Returns
    ---

    """
    src = rio.open(dem_file)
    dem = src.read(1)
    height, width = src.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rio.transform.xy(src.transform, rows, cols)
    X = np.array(xs)
    Y = np.array(ys)
    x = X[0, :]
    y = Y[:, 0]
    return x, y, dem


def get_v_components(vel_file, vel_fill_file,
                     xmin, xmax,
                     ymin, ymax):
    """
    Grab components of ice velocity from ITS_LIVE data and fill in missing data.

    Parameters
    vel_fill_file : str
        Path to a velocity raster/array to fill in where the annual mosaics may be missing data (ITS_LIVE composite)
    vel_files : list of str
        List of paths to velocity rasters/arrays to reference. This is generally for using ITS_LIVE annual mosaics
    xmin : float

    xmax : float

    ymin : float

    ymax : float


    Returns
    ---
    x_vel : (ny,nx) array
       x-coordinate grid points
    y_vel : (ny,nx) array
        y-coordinate grid points
    vx :  (ny,nx) array
        x-component of ice velocity
    vy : (ny,nx) array
        y-component of ice velocity
    v : (ny,nx) array
        magnitude of ice velocity
    """
    # load in the annual composite
    v_data = xr.open_dataset(vel_file)
    # filter out not
    v_data = v_data.isel(y=((v_data.y >= ymin) & (v_data.y <= ymax)),
                         x=((v_data.x >= xmin) & (v_data.x <= xmax))
                         )

    x_vel = v_data.x.values
    y_vel = v_data.y.values
    vx = v_data.vx.values.copy()
    if len(vx.shape) > 2:
        vx = vx.reshape(vx.shape[1:])
    vy = v_data.vy.values.copy()
    if len(vy.shape) > 2:
        vy = vy.reshape(vy.shape[1:])

    v_fill = xr.open_dataset(vel_fill_file)
    v_fill = v_fill.isel(y=((v_fill.y >= ymin) & (v_fill.y <= ymax)),
                         x=((v_fill.x >= xmin) & (v_fill.x <= xmax))
                         )

    x_vel_fill = v_fill.x.values
    y_vel_fill = v_fill.y.values
    vx_fill = v_fill.vx.values.copy()
    if len(vx_fill.shape) > 2:
        vx_fill = vx_fill.reshape(vx_fill.shape[1:])
    vy_fill = v_fill.vy.values.copy()
    if len(vy_fill.shape) > 2:
        vy_fill = vy_fill.reshape(vy_fill.shape[1:])

    v = np.sqrt(vx ** 2 + vy ** 2)
    vx[v > 15e3] = np.nan
    vy[v > 15e3] = np.nan
    ix_nan = np.isnan(vx) | np.isnan(vy)

    XV, YV = np.meshgrid(x_vel, y_vel)
    pts = np.array(list(zip(YV.flatten(), XV.flatten())))
    if np.isnan(vx).any():
        vx_interp = RegularGridInterpolator(
            (np.flip(y_vel_fill), x_vel_fill),
            np.flipud(vx_fill),
            bounds_error=False, fill_value=np.nan, method='linear'
        )
        vx_fill_interp = vx_interp(pts).reshape(vx.shape)
        vx[np.isnan(vx)] = vx_fill_interp[np.isnan(vx)]

    if np.isnan(vy).any():
        vy_interp = RegularGridInterpolator(
            (np.flip(y_vel_fill), x_vel_fill),
            np.flipud(vy_fill),
            bounds_error=False, fill_value=np.nan, method='linear'
        )
        vy_fill_interp = vy_interp(pts).reshape(vy.shape)
        vy[np.isnan(vy)] = vy_fill_interp[np.isnan(vy)]

    # now fill any remaining nan values with interpolation
    vx = inpaint_nans(vx)
    vy = inpaint_nans(vy)
    v = np.sqrt(vx ** 2 + vy ** 2)

    return x_vel, y_vel, vx, vy, v


def advect_dem1(x1, y1, dem1, x_vel, y_vel, vx, vy, dt, t_inc):
    """"""
    N_delt = round(abs(dt / t_inc))
    XS, YS = np.meshgrid(x1, y1)
    xf_ = XS.flatten()
    yf_ = YS.flatten()
    pts_start = np.array(list(zip(yf_, xf_)))
    pts = pts_start.copy()
    interpolator_u = RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vx),
        bounds_error=False, fill_value=np.nan, method='linear'
    )
    interpolator_v = RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vy),
        bounds_error=False, fill_value=np.nan, method='linear'
    )

    print(' .. time stepping')
    for i in np.arange(1, N_delt + 1):
        interpu = interpolator_u(pts)
        interpv = interpolator_v(pts)
        pts[:, 1] += interpu * t_inc / 365.25
        pts[:, 0] += interpv * t_inc / 365.25
        print('time step: %i / %i (%i days of %i)' % (i, N_delt, i * t_inc, dt), end='\r')

    print('interpolating dem1 to dem2 grid points')
    # interpolate to the later DEM grid points
    XE = pts[:, 1].reshape(XS.shape)
    YE = pts[:, 0].reshape(YS.shape)
    X2, Y2 = np.meshgrid(x2, y2)

    print('calculating dem1_lagrangian and difference')
    dem1_lagrangian = griddata(np.fliplr(pts), dem1.flatten(), (X2, Y2), method='cubic')
    dh = dem2 - dem1_lagrangian

    return X2, Y2, dh, dem1_lagrangian


def multiyear_demadvect(vel_fill_file, vel_files,
                        xmin, xmax, ymin, ymax,
                        x1, y1, dem1, dem2,
                        start_date, dt, t_inc):
    """
    Advect one DEM forward in time for direct comparsion and differencing.
    Largely inspired by/taken from Phillip Arndt's code for lagrangian differencing of DEMS (https://github.com/fliphilipp/LagrangianDEMdifferencing). Reproduced here with minimal changes for my own use.

    Parameters
    ---
    vel_fill_file : str
        Path to a velocity raster/array to fill in where the annual mosaics may be missing data (ITS_LIVE composite)
    vel_files : list of str
        List of paths to velocity rasters/arrays to reference. This is generally for using ITS_LIVE annual mosaics
    xmin : float

    xmax : float

    ymin : float

    ymax : float

    x1 : (nx,1) array
        x-coordinates of heights (h1) at start date
    y1 : (ny,1) array
        y-coordinates of heights (h1) at start date
    dem1 : (ny,nx) array
        heights (or whatever value) at start date
    dem2 : (ny, nx) array
        heights (or whatever value) at end date
    start_date : datetime.datetime object
        Date of x1,y1, and h1 data
    dt : float
        Total time to advect x1,y1, and h1
    t_inc : float
        Time steps of advection in days. Lower values are more stable

    Returns
    ---
    X2 : (ny,nx) array
        Grid points of x1,y1 after advection
    Y2 : (ny,nx) array
        Grid points of x1,y1 after advection
    dh : (ny,nx) array
        Lagrangian difference of DEM1 and
    dem1_lagrangian :
        Lagrangian advected input DEM (dem1)
    """

    N_delt = round(abs(dt / t_inc))
    XS, YS = np.meshgrid(x1, y1)
    xf_ = XS.flatten()
    yf_ = YS.flatten()
    pts_start = np.array(list(zip(yf_, xf_)))
    pts = pts_start.copy()

    start_year = start_date.year
    if start_year in vel_files:
        yr = start_year
    elif start_year < min(vel_files):
        yr = min(vel_files)
    elif start_year > max(vel_files):
        yr = max(vel_files)
    x_vel, y_vel, vx, vy, v = get_v_components(vel_files[yr], vel_fill_file,
                                               xmin, xmax,
                                               ymin, ymax
                                               )
    interpolator_u = RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vx),
        bounds_error=False, fill_value=np.nan, method='linear'
    )
    interpolator_v = RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vy),
        bounds_error=False, fill_value=np.nan, method='linear'
    )

    for i in np.arange(1, N_delt + 1):
        interpu = interpolator_u(pts)
        interpv = interpolator_v(pts)
        pts[:, 1] += interpu * t_inc / 365.25
        pts[:, 0] += interpv * t_inc / 365.25
        start_date = start_date + datetime.timedelta(days=t_inc)
        if start_date.year > start_year:
            print(f'new year, transitioning from {start_year} = > {start_date.year}')
            start_year = start_date.year
            print('   grabbing that years stuff')
            if start_year in vel_files:
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[start_year], vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax
                                                           )
            elif start_year > max(vel_files):
                yr = max(vel_files)
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[yr], vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax
                                                           )
            elif start_year < min(vel_files):
                yr = min(vel_files)
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[yr], vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax
                                                           )
            print('   and interpolating')
            interpolator_u = RegularGridInterpolator(
                (np.flip(y_vel), x_vel), np.flipud(vx),
                bounds_error=False, fill_value=np.nan, method='linear'
            )
            interpolator_v = RegularGridInterpolator(
                (np.flip(y_vel), x_vel), np.flipud(vy),
                bounds_error=False, fill_value=np.nan, method='linear'
            )

        print('time step: %i / %i (%i days of %i)' % (i, N_delt, i * t_inc, dt))

    print('interpolating dem1 to dem2 grid points')
    # interpolate to the later DEM grid points
    # XE = pts[:, 1].reshape(XS.shape)
    # YE = pts[:, 0].reshape(YS.shape)
    X2, Y2 = np.meshgrid(x2, y2)

    print('calculating/interpolating dem1_lagrangian')
    dem1_lagrangian = griddata(np.fliplr(pts), dem1.flatten(), (X2, Y2), method='cubic')
    print('now calculating dem2 - dem1_lagrangian')
    dh = dem2 - dem1_lagrangian

    return X2, Y2, dh, dem1_lagrangian


def sample_along_profile(ds, value, x, y, method='nearest'):
    """
    Function to automate sampling along an arbitrary profile as xarray requires selecting data only by row and column. Using

    Parameters
    ---
    ds : xarray.Dataset, xarray.DataArray
        Xarray Dataset or DataArray with coordinates 'x' and 'y' and variable 'value'.
    value : str
    x : np.ndarray, list
        List or numpy array of values in the x-coordinate
    y : np.ndarray, list
        List or numpy array of values in the y-coordinate
    method : str
        Method with which to sample the data (see xarray.Dataset.sel for explanation). Default is 'nearest'

    Returns
    ---
    along_profile : np.ndarray
        Sampling of 'value' along the arbitrary line (x,y)

    """
    assert value in ds.variables
    return np.array([ds.sel(x=xi, y=yi, method=method)[value] for (xi, yi) in zip(x, y)])
