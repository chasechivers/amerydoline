import os
import pickle
import re

import numpy as np
import pyproj
import shapefile
import shapely


def load_pkl(filename):
    """
     Load .pkl file into memory.

     Parameters
     ---
     filename : str
         Path and filename for object to be saved

     Returns
     ---
     ret : object
        Loaded pkl file
     """
    with open(filename, 'rb') as inp:
        ret = pickle.load(inp)
    return ret


def save_as_pkl(filename, obj):
    """
    Save object as a pkl file

    Parameters
    ---
    filename : str
        Path and filename for object to be saved

    Returns
    ---
    None
    """
    with open(filename, 'wb') as out:
        pickle.dump(obj, out)


def set_hemisphere(GRANULE):
    """
    Taken from T. Sutterly's ICESat-2 tools: https://github.com/tsutterley/read-ICESat-2
    """
    if GRANULE in ('10', '11', '12'):
        projection_flag = 'S'
    elif GRANULE in ('03', '04', '05'):
        projection_flag = 'N'
    return projection_flag


def get_IS2_transformer(HEM='S'):
    """
    Get pyproj function to transform EPSG4326 (lon, lat) to EPSG3031 (x,y).

    Taken from T. Sutterly's ICESat-2 tools: https://github.com/tsutterley/read-ICESat-2

    Parameters
    ---
    HEM : str (default = 'S')
        Either 'N' for North (EPSG3413) or 'S' for south (EPSG3031)

    Returns
    ---
    transformer : function
    """
    EPSG = dict(N=3413, S=3031)
    crs1 = pyproj.CRS.from_epsg(4326)
    crs2 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    return transformer


def reverse_transform(HEM='S'):
    """
    Get pyproj function to transform EPSG3031 (x,y) to EPSG4326 (lon, lat)

    Taken from T. Sutterly's ICESat-2 tools: https://github.com/tsutterley/read-ICESat-2

    Parameters
    ---
    HEM : str (default = 'S')
        Either 'N' for North (EPSG3413) or 'S' for south (EPSG3031)

    Returns
    ---
    transformer : function
    """
    EPSG = dict(N=3413, S=3031)
    crs2 = pyproj.CRS.from_epsg(4326)
    crs1 = pyproj.CRS.from_epsg(EPSG[HEM])
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    return transformer


def ll2xy(lon, lat):
    """
    Transform EPSG4326 (lon, lat) values to EPSG3031 (x,y) values

    Parameters
    ---
    longitude : np.ndarray
        longitude values in
    latitude : np.ndarray
        latitude values in EPSG4326

    Returns
    ---
    x : list, np.ndarray
        x-coordinates in EPSG3031
    y : list, np.ndarray
        y-coordinates in EPSG3031

    """
    return get_IS2_transformer('S').transform(lon, lat)


def xy2ll(x, y):
    """
    Transform EPSG3031 (x,y) values to EPSG4326 (lon, lat) values

    Parameters
    ---
    x : list, np.ndarray
        x-coordinates in EPSG3031
    y : list, np.ndarray
        y-coordinates in EPSG3031

    Returns
    ---
    longitude : np.ndarray
        longitude values in EPSG4326
    latitude : np.ndarray
        latitude values in EPSG4326
    """
    return reverse_transform('S').transform(x, y)


def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500000):
    """
    Returns list of paths to files given an input_dir, path_pattern, and file_pattern

    Parameters
    ---
    input_dir : str
        Path to directory to search
    path_pattern : str
        Paths to files MUST include this path
    file_pattern : str
        Paths to files MUST include this as part of the file name
    maxResults : int

    Returns
    ---
    file_paths : list
        List of paths that meet the criteria
    """
    file_paths = []
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Could not find path: {input_dir}")
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if re.search(path_pattern, dirpath):
            file_list = [item for item in filenames if re.search(file_pattern, item)]
            file_path_list = [os.path.join(dirpath, item) for item in file_list]
            file_paths += file_path_list
            if len(file_paths) > maxResults:
                break
    return file_paths[0:maxResults]


def create_polydict_ptsarr(shpfilename):
    """
    Dictionary of shapefile objects

    Parameters
    ---
    shpfilename : str
        Path to .shp/.shx file that has multiple polygons, lines, etc.

    Returns
    ---
    poly_dict : dict
        Dictionary of shp objects indexed by their index in the shapefile
    """
    shpfile = shapefile.Reader(shpfilename)
    shape_entities = shpfile.shapes()
    shape_attributes = shpfile.records()

    poly_dict = {}

    indices = [i for i, a in enumerate(shape_attributes)]

    for i in indices:
        points = np.array(shape_entities[i].points)
        parts = shape_entities[i].parts
        parts.append(len(points))
        poly_list = []
        for p1, p2 in zip(parts[:-1], parts[1:]):
            poly_list.append(list(zip(points[p1:p2, 0], points[p1:p2, 1])))
        poly_obj = shapely.geometry.Polygon(poly_list[0], poly_list[1:])
        poly_dict[shape_attributes[i][0]] = poly_obj.buffer(0)

    return poly_dict, points


def does_this_intersect(poly_dict, x, y):
    """
    Determines whether a line (x,y) intersects with a polygon shapely object.

    Parameters
    ---
    poly_dict : dict of shapely polygon objects
    x : np.ndarray, list
        Array of x-coordinate values (shared CRS with polygons)
    y : np.ndarray, list
        Array of y-coordinate values (shared CRS with polygons)

    Returns
    ---
    intersects : bool
    """

    xy_point = shapely.geometry.MultiPoint(np.c_[x, y])
    tests = []
    for key, poly_obj in poly_dict.items():
        int_test = poly_obj.intersects(xy_point)
        tests.append(int_test)
    intersects = np.any(tests)
    return intersects


def sample_raster(x, y, rio_obj, fillval=-9999, mask=False):
    """
    Function to sample a raster along an arbitrary line.

    Parameters
    ---
    x : np.ndarray, list
        List or numpy array of values in the x-coordinate of the CRS in rio_obj
    y : np.ndarray, list
        List or numpy array of values in the y-coordinate of the CRS in rio_obj
    rio_obj : rasterio raster object
        Raster that want to sample from along (x,y)
    fillval : float
    mask : bool

    Returns
    ---
    x : np.ndarray
        x-coordinate
    y : np.ndarray
        y-coordinates
    val : np.npdarray
        Values of rio_obj along (x,y)
    """
    try:
        len(x)
    except:
        x = x + np.zeros(len(y))
    try:
        len(y)
    except:
        y = y + np.zeros(len(x))
    sample_coords = [(xi, yi) for (xi, yi) in zip(x, y)]
    val = np.squeeze(np.array([v for v in rio_obj.sample(sample_coords)]))
    if mask:
        maskvals = val <= fillval
        val[maskvals] = np.nan
        maskvals = ~np.isnan(val)
        x = x[maskvals]
        y = y[maskvals]
        val = val[maskvals]
    return x, y, val


def sample_along_profile(ds, value, x, y, method='nearest'):
    """
    Function to automate sampling along an arbitrary profile of an array using xarray, as xarray requires selecting data only by row and column.

    Parameters
    ---
    ds : xarray.Dataset, xarray.DataArray
        Xarray Dataset or DataArray with coordinates 'x' and 'y' and variable 'value'.
    value : str
        Name of array/data to sample from.
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

def read_ATL06_alt(file):
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
