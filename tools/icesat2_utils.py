import datetime

import h5py
import numpy as np
from scipy import interpolate

from tools.dem_utils import get_v_components
from tools.utils import load_pkl, ll2xy


def read_ATL06_alt(file):
    """
    Modified from https://github.com/tsutterley/read-ICESat-2/blob/main/icesat2_toolkit/ Although the majority is extracted exactly from there, I have modified it to serve my own purposes.


    """
    bstr = 'gt{0}{1}'
    # beam pairs
    pair = [1, 2, 3]
    # left/right of beam pair
    lr = ['l', 'r']
    beams = [bstr.format(a, b) for (a, b) in zip(pair * len(lr), lr * len(pair))]

    list_of_keys = ['atl06_quality_summary', 'delta_time', 'h_li', 'h_li_sigma',
                    'latitude', 'longitude', 'segment_id', 'sigma_geo_h',
                    'ground_track']

    data = h5py.File(file, 'r')

    ans = {bstr.format(kP, ''): {kK: {} for kK in list_of_keys} for kP in pair}
    for kP in range(len(pair)):
        ID = {}
        for kB in range(len(lr)):
            ID[kB] = data[bstr.format(kP + 1, lr[kB])]['land_ice_segments'][
                'segment_id']
        ID_both = np.unique(np.hstack([v for k, v in ID.items()]))
        n_seg = ID_both.size
        in_ind = {}
        out_ind = {}
        for kB in range(len(lr)):
            _, in_ind[kB], out_ind[kB] = np.intersect1d(ID[kB], ID_both,
                                                        return_indices=True
                                                        )
        for kK in range(len(list_of_keys)):
            K = bstr.format(kP + 1, '')
            key = list_of_keys[kK]
            tmp = {}
            for kB in range(len(lr)):
                b = bstr.format(kP + 1, lr[kB])
                if key in ['ground_track']:
                    for xx in ['x_atc', 'y_atc']:
                        arr = np.copy(data[b]['land_ice_segments'][key][xx])
                        fv = data[b]['land_ice_segments'][key][xx].fillvalue
                        if fv != 0:
                            arr[arr == fv] = np.nan
                        tmp[xx] = arr
                else:
                    arr = np.copy(data[b]['land_ice_segments'][key])
                    fv = data[b]['land_ice_segments'][key].fillvalue
                    if fv != 0:
                        arr[arr == fv] = np.nan
                    tmp[kB] = arr

            if key in ['ground_track']:
                for xx in ['x_atc', 'y_atc']:
                    ans[K][xx] = np.nan * np.zeros((2, n_seg))
                    for kB in range(len(lr)):
                        ans[K][xx][kB, out_ind[kB]] = tmp[xx][in_ind[kB]]
                _ = ans[K].pop(key)
            else:
                ans[K][key] = np.nan * np.zeros((2, n_seg))
                for kB in range(len(lr)):
                    ans[K][key][kB, out_ind[kB]] = tmp[kB][in_ind[kB]]
    return ans


def clean_nans(x, y, h):
    """
    Cleans the nan values from (2, n) ATL06 ararys.

    Parameters
    ---
    x : (2, n) np.ndarray
        where 2 is the number of beams (left, right) is and n is the number of points in space
    y : (2, n) np.ndarray
        where 2 is the number of beams (left, right)  and n is the number of points in space
    h : (2, n) np.ndarray
        ICESat-2 heights from ATL06 (or anything really..) where 2 is the number of beams (left, right)  and n is the number of points in space

    Returns
    ---
    x : (m, l) np.ndarray
    y : (m, l) np.ndarray
    h : (m, l) np.ndarray

    """
    num_beams, n = x.shape
    _x = []
    _y = []
    _h = []

    for beam in range(num_beams):
        ix = ~np.isnan(x[beam]) & ~np.isnan(y[beam])
        _x.append(x[beam, ix])
        _y.append(y[beam, ix])
        _h.append(h[beam, ix])
    x, y, h = np.array(_x), np.array(_y), np.array(_h)

    if x.dtype == object:
        n1 = len(x[0])
        n2 = len(x[1])
        n = np.min([n1, n2])
        x = np.vstack([x[0][:n], x[1][:n]])
        y = np.vstack([y[0][:n], y[1][:n]])
        h = np.vstack([h[0][:n], h[1][:n]])

    return x, y, h


def advect(x1, y1, h1, x2, y2,
           vel_files, vel_fill_file,
           start_date, dt, t_inc,
           xmin, xmax, ymin, ymax):
    """
    Function for only the actual advection calculations of x1,y1, and h1. Essentially the same as advect_dem1 in tools/dem_utils.py.
    Largely inspired by/taken from Phillip Arndt's code for lagrangian differencing of DEMS (https://github.com/fliphilipp/LagrangianDEMdifferencing). Reproduced here with minimal changes for my own use.

    Parameters
    ---
    x1 : (n,1) array
        x-coordinates of heights (h1) at start date
    y1 : (n,1) array
        y-coordinates of heights (h1) at start date
    h1 : (n,1) array
        heights (or whatever value) at start date
    x2 : (n,1) array

    y2 : (n,1) array

    vel_files : list of str
        List of paths to velocity rasters/arrays to reference. This is generally for using ITS_LIVE annual mosaics
    vel_fill_file : str
        Path to a velocity raster/array to fill in where the annual mosaics may be missing data (ITS_LIVE composite)
    start_date : datetime.datetime object
        Date of x1,y1, and h1 data
    dt : float
        Total time to advect x1,y1, and h1
    t_inc : float
        Time steps of advection in days. Lower values are more stable
    xmin : float
    xmax : float
    ymin : float
    ymax : float

    Returns
    ---
    x1l : array
        Lagrangian advected x1 array
    y1l : array
        Lagrangian advected y1 array
    h1_lagrangian : array
        Lagrangian advected heights (h1)
    """
    pts_start = np.array([[yi, xi] for (xi, yi) in zip(x1, y1)])
    pts = pts_start.copy()

    max_year = max(vel_files)
    min_year = min(vel_files)
    start_year = start_date.year
    if start_year > max_year:
        start_year = max_year

    x_vel, y_vel, vx, vy, v = get_v_components(vel_files[start_year],
                                               vel_fill_file,
                                               xmin, xmax,
                                               ymin, ymax)

    vx_interp = interpolate.RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vx),
        bounds_error=False, fill_value=np.nan, method='linear'
    )
    vy_interp = interpolate.RegularGridInterpolator(
        (np.flip(y_vel), x_vel), np.flipud(vy),
        bounds_error=False, fill_value=np.nan, method='linear'
    )

    N_delt = round(abs(dt / t_inc))

    for i in np.arange(1, N_delt + 1):
        vxi = vx_interp(pts)
        vyi = vy_interp(pts)
        pts[:, 1] += vxi * t_inc / 365.25
        pts[:, 0] += vyi * t_inc / 365.25
        start_date = start_date + datetime.timedelta(days=t_inc)
        if start_date.year > start_year:
            start_year = start_date.year
            if start_year in vel_files:
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[start_year],
                                                           vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax)
            elif start_year > max(vel_files):
                yr = max_year
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[yr],
                                                           vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax)
            elif start_year < min(vel_files):
                yr = min_year
                x_vel, y_vel, vx, vy, v = get_v_components(vel_files[yr],
                                                           vel_fill_file,
                                                           xmin, xmax,
                                                           ymin, ymax)

            vx_interp = interpolate.RegularGridInterpolator(
                (np.flip(y_vel), x_vel), np.flipud(vx),
                bounds_error=False, fill_value=np.nan, method='linear'
            )
            vy_interp = interpolate.RegularGridInterpolator(
                (np.flip(y_vel), x_vel), np.flipud(vy),
                bounds_error=False, fill_value=np.nan, method='linear'
            )
        # print('time step: %i / %i (%i days of %i)' % (i, N_delt, i * t_inc, dt))

    ix = ~np.isnan(pts[:, 0])
    x1l, y1l = pts[ix, 1], pts[ix, 0]
    pts_clean = np.c_[pts[ix, 0], pts[ix, 1]]
    h1_clean = h1[ix]

    h1_lagrangian = interpolate.griddata(np.fliplr(pts_clean), h1_clean, (x2, y2),
                                         method='nearest')

    return x1l, y1l, h1_lagrangian


def advect_IS2(IS2_file1, IS2_file2,
               vel_files, vel_fill_file,
               start_date, dt, t_inc=1,
               xmin=1838747.0, xmax=1849155.0,
               ymin=681349.0, ymax=691219.0,
               gt=None):
    """
    Advect ICESat-2 elevations forward in time along ice flow for direct comparison. Not advised for qualitative comparison.
    Largely inspired by/taken from Phillip Arndt's code for lagrangian differencing of DEMS (https://github.com/fliphilipp/LagrangianDEMdifferencing). Reproduced here with minimal changes for my own use.

    Parameters
    ---
    IS2_file1 : str
        Location of IS2 file whose heights (h_li) you would like to advect to  IS2_file2
    IS2_file2 : str
        Location of IS2 files whose heights (h_li) you would like to be the reference time for advection
    vel_files : list
        Dictionary of the location of ITS_LIVE Antarctica annual velocity composites files in the netCDF format structured by year
        e.g. vel_files = {2014:'./data/ANT_G0240_2014.nc', 2016:'./data/ANT_G0240_2016.nc'}
    vel_fill_file : str
        Location of the ITS_LIVE Antarctica total composite ANT_G0120_0000.nc
    dt : float
        Difference in time between the IS2_file1 data and IS2_file2 data in days
    t_inc : float (default = 1 day)
        Time step during advection in days.
    xmin : float
    xmax : float
    ymin : float
    ymax : float

    Returns
    ---
    x1l : np.ndarray
    y1l : np.ndarray
    x2l : np.ndarray
    y2l : np.ndarray
    h1l_lagrangian : np.ndarray
    h2l : np.ndarray
    x1r : np.ndarray
    y1r : np.ndarray
    x2r : np.ndarray
    y2r : np.ndarray
    h1r_lagrangian : np.ndarray
    h2r : np.ndarray


    """

    IS2_data1 = load_pkl(IS2_file1)
    if gt is None:
        gtx = np.where(['gt1' in IS2_data1.keys(),
                        'gt2' in IS2_data1.keys(),
                        'gt3' in IS2_data1.keys()])[0][0]
        gt = ['gt1', 'gt2', 'gt3'][gtx]
    IS2_data1 = IS2_data1[gt]['masked']['Amery']

    IS2_data2 = load_pkl(IS2_file2)
    IS2_data2 = IS2_data2[gt]['masked']['Amery']

    lon1, lat1, h1 = clean_nans(IS2_data1['longitude'],
                                IS2_data1['latitude'],
                                IS2_data1['h_li'])
    lon2, lat2, h2 = clean_nans(IS2_data2['longitude'],
                                IS2_data2['latitude'],
                                IS2_data2['h_li'])

    x1, y1 = np.zeros(lon1.shape), np.zeros(lat1.shape)
    x2, y2 = np.zeros(lon2.shape), np.zeros(lat2.shape)
    for beam in [0, 1]:
        x1[beam, :], y1[beam, :] = ll2xy(lon1[beam, :], lat1[beam, :])
        x2[beam, :], y2[beam, :] = ll2xy(lon2[beam, :], lat2[beam, :])

    # start filtering, but must do this by beam
    x1l, y1l, h1l = x1[0, :], y1[0, :], h1[0, :]
    x1r, y1r, h1r = x1[1, :], y1[1, :], h1[1, :]
    x2l, y2l, h2l = x2[0, :], y2[0, :], h2[0, :]
    x2r, y2r, h2r = x2[1, :], y2[1, :], h2[1, :]

    ix = ((x1l >= xmin) & (x1l <= xmax)) & \
         ((y1l >= ymin) & (y1l <= ymax))
    x1l = x1l[ix]
    y1l = y1l[ix]
    h1l = h1l[ix]

    ix = ((x1r >= xmin) & (x1r <= xmax)) & \
         ((y1r >= ymin) & (y1r <= ymax))
    x1r = x1r[ix]
    y1r = y1r[ix]
    h1r = h1r[ix]

    ix = ((x2l >= xmin) & (x2l <= xmax)) & \
         ((y2l >= ymin) & (y2l <= ymax))
    x2l = x2l[ix]
    y2l = y2l[ix]
    h2l = h2l[ix]

    ix = ((x2r >= xmin) & (x2r <= xmax)) & \
         ((y2r >= ymin) & (y2r <= ymax))
    x2r = x2r[ix]
    y2r = y2r[ix]
    h2r = h2r[ix]

    # advect the left beam
    _, _, h1l_lagrangian = advect(x1l, y1l, h1l, x2l, y2l,
                                  vel_files, vel_fill_file,
                                  start_date, dt, t_inc,
                                  xmin, xmax, ymin, ymax)
    # advect the right beam
    _, _, h1r_lagrangian = advect(x1r, y1r, h1r, x2r, y2r,
                                  vel_files, vel_fill_file,
                                  start_date, dt, t_inc,
                                  xmin, xmax, ymin, ymax)

    return x1l, y1l, x2l, y2l, h1l_lagrangian, h2l, x1r, y1r, x2r, y2r, h1r_lagrangian, h2r
