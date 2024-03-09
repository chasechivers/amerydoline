import copy
import itertools
import shutil
import time as timer

import dask
import fiona
import h5py
from dask.distributed import Client
from shapely.geometry import MultiPoint, Point, Polygon, mapping

from tools.utils import *

# choose number of processors for dask to use
N_WORKERS = 2
base_dir = '/'

################################################################################################
# directories + files
################################################################################################

# directory where to save filtered and masked files
# save_dir = 'Amery Doline Project/data/tmp'
# save_dir = 'AmeryDolineProject/data/ICESat-2/MASKED'
# tmp_dir = 'AmeryDolineProject/data/ICESat-2/tmp'
save_dir = '/data/ICESat-2'
tmp_dir = save_dir + '/tmp'

# Point to where all the .h5 ICESat-2 data is
IS2_data_dir = save_dir

# Mask the ICESat-2 tracks to the Amery ice shelf file
iceshelf_filename = 'GIS/AmeryMappingBoundingBox.shp'

# Filter out ICESat-2 tracks that don't intersect with this shapefile
shp_filename = 'IS/DolineOutlines.shp'
# shp_filename = iceshelf_filename

# append base path to filename
shp_filename = os.path.join(base_dir, shp_filename)

data_dir = os.path.join(base_dir, IS2_data_dir)
save_dir = os.path.join(base_dir, save_dir)
shp_save_dir = os.path.join(save_dir, 'shp')

data_list = directory_spider(data_dir, file_pattern='.h5')
# exclude these TRK in case others are available -OR- only include these types
# data_list = [f for f in data_list if '_0843' not in f and '_0470' not in f and
#             '_0401' not in f]
# data_list = [f for f in data_list if '2019' in f and '_0843' in f]

HEM = 'S'
transformer = get_IS2_transformer(HEM)
pair_beam = list(itertools.product(['gt1', 'gt2', 'gt3'], [0, 1]))
shp_dict, _ = create_polydict_ptsarr(shp_filename)


################################################################################################
# Some useful functions
################################################################################################

def read_in_data(file):
    # read in .h5 file in H5PY format
    hf = h5py.File(file, 'r')
    # read in .h5 file as a dictionary with only the relevant info
    dic = read_ATL06_alt(file)
    return hf, dic


def load_ice_shelf(shpfile, shelf_filter=None, BUFFER=0.):
    if shelf_filter is None:
        shelf_filter = ''
    shape_input = shapefile.Reader(shpfile)
    # reading regional shapefile
    shape_entities = shape_input.shapes()
    shape_attributes = shape_input.records()

    # python dictionary of polygon objects
    poly_dict = {}

    # iterate through shape entities and attributes to find FL attributes
    indices = [i for i, a in enumerate(shape_attributes) if (a[2] == 'FL' and
                                                             a[0] in shelf_filter)]

    # for each floating ice indice
    for i in indices:
        # extract Polar-Stereographic coordinates for record
        points = np.array(shape_entities[i].points)
        # shape entity can have multiple parts
        parts = shape_entities[i].parts
        parts.append(len(points))
        # list object for x,y coordinates (exterior and holes)
        poly_list = []
        for p1, p2 in zip(parts[:-1], parts[1:]):
            poly_list.append(list(zip(points[p1:p2, 0], points[p1:p2, 1])))
        # convert poly_list into Polygon object with holes
        poly_obj = Polygon(poly_list[0], poly_list[1:])
        # buffer polygon object and add to total polygon dictionary object
        poly_dict[shape_attributes[i][0]] = poly_obj.buffer(BUFFER * 1e3)

    # return the polygon object and the input file name
    return poly_dict


def get_mask_per_shelf(xy_point, masked, poly_dict, n_seg, pair, beam):
    # calculate mask for each ice shelf
    boolarr = np.zeros(n_seg, dtype=bool)
    associated_map = {}
    i2b = ['l', 'r']
    if 'ice_shelf_mask' not in masked[pair].keys():
        masked[pair]['ice_shelf_mask'] = {}
        print(f'key ice_shelf_mask in masked[pa'
              f'ir]: {"ice_shelf_mask" in masked[pair].keys()}')
    for shelf, poly_obj in poly_dict.items():
        print('>>>shelf :', shelf)
        int_test = poly_obj.intersects(xy_point)
        print('>>> interescts with shelf:', int_test)
        if int_test:
            if shelf not in masked[pair]['ice_shelf_mask']:
                print('shelf is not in masked[pair]')
                masked[pair]['ice_shelf_mask'][shelf] = np.zeros((2, n_seg),
                                                                 dtype=bool
                                                                 )

            distributed_map = boolarr.copy()
            associated_map[shelf] = boolarr.copy()
            int_map = list(map(poly_obj.intersects,
                               list(map(xy_point.geoms._get_geom_item, range(n_seg)))
                               )
                           )

            int_indices, = np.nonzero(int_map)
            distributed_map[int_indices] = True
            associated_map[shelf] = copy.copy(distributed_map)

    for shelf in associated_map:
        print(associated_map[shelf])
        masked[pair]['ice_shelf_mask'][shelf][beam, :] = np.bool_(associated_map[shelf])
    return masked


def mask_arrs(masked):
    for pair in masked:
        if 'ice_shelf_mask' not in masked[pair]:
            pass
        else:
            masked[pair]['masked'] = {}
            for shelf in masked[pair]['ice_shelf_mask']:
                masked[pair]['masked'][shelf] = {}
                for val in masked[pair]:
                    if val not in ['ice_shelf_mask', 'masked']:
                        mask = np.bool_(masked[pair]['ice_shelf_mask'][shelf])
                        masked[pair]['masked'][shelf][val] = np.nan * np.zeros(mask.shape)
                        for beam in [0, 1]:
                            masked[pair]['masked'][shelf][val][beam, mask[beam]] = masked[
                                pair][
                                val][
                                beam, mask[beam]]
    return masked


# slightly different implementation of tools.utils function
def does_this_intersect(poly_dict, lon, lat):
    xy_point = shapely.geometry.MultiPoint(np.c_[lon, lat])
    tests = []
    for key, poly_obj in poly_dict.items():
        poly_obj = poly_obj.buffer(2e3)
        int_test = poly_obj.intersects(xy_point)
        tests.append(int_test)
    return np.any(tests)


def write_masked_pkl(masked, file, OUT_DIR):
    start = file.find('ATL')
    PRD, YY, MM, DD, HH, MN, SS, TRK, CYC, GRN, RL, VRS, AUX = rx.findall(
        file[start:]
    ).pop()
    fargs = (
        PRD, "MASKED", YY, MM, DD, HH, MN, SS, TRK, CYC, GRN, RL, VRS, AUX)
    file_format = '{0}_{1}_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.pkl'

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(os.path.join(OUT_DIR, YY)):
        os.mkdir(os.path.join(OUT_DIR, YY))

    if not os.path.exists(os.path.join(OUT_DIR, YY, TRK)):
        os.mkdir(os.path.join(OUT_DIR, YY, TRK))

    OUT_DIR = os.path.join(OUT_DIR, YY, TRK)

    filename = os.path.join(OUT_DIR, file_format.format(*fargs))
    print('>>> !!! write_pkl: saving as', filename)
    save_as_pkl(filename, masked)
    return masked


def write_masked_shp(masked, file, OUT_DIR):
    start = file.find('ATL')
    PRD, YY, MM, DD, HH, MN, SS, TRK, CYC, GRN, RL, VRS, AUX = rx.findall(
        file[start:]
    ).pop()

    file_format = '{0}_{1}_{2}_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.shp'
    i2b = ['l', 'r']
    schema = {
        'geometry': 'Point', 'properties':
            {'id': 'int', 'elevation (m)': 'float'}
    }

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(os.path.join(OUT_DIR, YY)):
        os.mkdir(os.path.join(OUT_DIR, YY))

    if not os.path.exists(os.path.join(OUT_DIR, YY, TRK)):
        os.mkdir(os.path.join(OUT_DIR, YY, TRK))

    OUT_DIR = os.path.join(OUT_DIR, YY, TRK)

    for pair in masked:
        if 'masked' in masked[pair]:
            for shelf in masked[pair]['masked']:
                print(f'{pair}{shelf}')
                for beam in [0, 1]:
                    fargs = (
                        PRD, f"{shelf}_MASKED", f"{pair}{i2b[beam]}", YY, MM, DD, HH, MN,
                        SS,
                        TRK, CYC, GRN, RL, VRS, AUX)
                    filename = os.path.join(OUT_DIR, file_format.format(*fargs))
                    print(f'>>>write_shp: saving as {filename}')
                    lat = masked[pair]['masked'][shelf]['latitude'][beam, :]
                    lon = masked[pair]['masked'][shelf]['longitude'][beam, :]

                    elev = masked[pair]['masked'][shelf]['h_li'][beam]
                    nanfilter = np.logical_and(~np.isnan(elev),
                                               np.logical_and(~np.isnan(lon),
                                                              ~np.isnan(lat)
                                                              )
                                               )
                    x, y = transformer.transform(lon[nanfilter], lat[nanfilter])
                    points = [Point(xi, yi) for (xi, yi) in zip(x, y)]
                    elev = elev[nanfilter]
                    with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as layer:
                        for i, point in enumerate(points):
                            elem = {}
                            elem['geometry'] = mapping(point)
                            elem['properties'] = {
                                'id': i + 1,
                                'elevation (m)': float(elev[i])
                            }
                            layer.write(elem)
                    layer.close()
    return masked


def write_masked_shp_line(masked, file, OUT_DIR):
    start = file.find('ATL')
    PRD, YY, MM, DD, HH, MN, SS, TRK, CYC, GRN, RL, VRS, AUX = rx.findall(
        file[start:]
    ).pop()

    file_format = '{0}_{1}_{2}_{3}{4}{5}{6}{7}{8}_{9}{10}{11}_{12}_{13}{14}.shp'
    i2b = ['l', 'r']
    schema = {
        'geometry': 'Line', 'properties':
            {'id': 'int', 'track': 'float'}
    }

    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(os.path.join(OUT_DIR, YY)):
        os.mkdir(os.path.join(OUT_DIR, YY))

    if not os.path.exists(os.path.join(OUT_DIR, YY, TRK)):
        os.mkdir(os.path.join(OUT_DIR, YY, TRK))

    OUT_DIR = os.path.join(OUT_DIR, YY, TRK)

    print('>>write_masked_shp')
    for pair in masked:
        print('>>>write_masked_shp', pair)
        if 'masked' in masked[pair]:
            for shelf in masked[pair]['masked']:
                print(f'{pair}{shelf}')
                for beam in [0, 1]:
                    fargs = (
                        PRD, f"{shelf}_MASKED", f"{pair}{i2b[beam]}", YY, MM, DD, HH, MN,
                        SS,
                        TRK, CYC, GRN, RL, VRS, AUX)
                    filename = os.path.join(OUT_DIR, file_format.format(*fargs))
                    lat = masked[pair]['masked'][shelf]['latitude'][beam, :]
                    lon = masked[pair]['masked'][shelf]['longitude'][beam, :]

                    elev = masked[pair]['masked'][shelf]['h_li'][beam]
                    nanfilter = np.logical_and(~np.isnan(elev),
                                               np.logical_and(~np.isnan(lon),
                                                              ~np.isnan(lat)
                                                              )
                                               )

                    x, y = transformer.transform(lon[nanfilter], lat[nanfilter])
                    points = [Point(xi, yi) for (xi, yi) in zip(x, y)]
                    elev = elev[nanfilter]
                    with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as layer:
                        for i, point in enumerate(points):
                            elem = {}
                            elem['geometry'] = mapping(point)
                            elem['properties'] = {
                                'id': i + 1,
                                'elevation (m)': float(elev[i])
                            }
                            layer.write(elem)
                    layer.close()
    return masked


def filter_mask_write(file):
    print(f'loading {file[-39:]}')
    hf, data = read_in_data(file)
    print('>>> finished reading')
    masked = copy.copy(data)
    print('>>> copied data')
    any_intersects = False
    for (pair, beam) in pair_beam:
        print(f'>>> >>> {file[-39:]}{pair}{beam}')
        n_seg = data[pair]['segment_id'].shape[1]
        print(f'>>> >>> {file[-39:]}{pair}{beam}: grabbing lat lon')
        longitude = data[pair]['longitude'][beam, :].copy()
        latitude = data[pair]['latitude'][beam, :].copy()
        print(f'>>> >>> {file[-39:]}{pair}{beam}: filtering out nans')
        nanfilter = np.logical_and(~np.isnan(longitude), ~np.isnan(latitude))
        print(f'>>> >>> {file[-39:]}{pair}{beam}: transforming to EPGS3031')
        X, Y = transformer.transform(longitude, latitude)
        print(f'>>> >>> {file[-39:]}{pair}{beam}: determining intersections')
        intersects_shpfile = does_this_intersect(shp_dict, X[nanfilter], Y[nanfilter])
        print(f'>>> {file[-39:]} intersects? {intersects_shpfile}')
        if intersects_shpfile:
            any_intersects = True
            print(f'>>> {file[-39:]}:{pair}{beam} intersects with {shp_filename[:-18]}')

            # convert reduced x and y to MultiPoint object
            xy_point = MultiPoint(np.c_[longitude, latitude])
            print(f'>>> masking {file[-39:]},{pair} {beam}')
            masked = get_mask_per_shelf(xy_point, masked, iceshelf_dict, n_seg, pair,
                                        beam
                                        )
        else:
            pass
    if any_intersects:
        print('>>> !! THIS DOES INTERSECT')
        masked_ = mask_arrs(masked)
        # filter out tracks that don't intersect for smaller file sizes
        masked = {}
        for k in masked_:
            if 'masked' in masked_[k]:
                masked[k] = masked_[k]
        print('>>> WRITING TO PKL???')
        masked = write_masked_pkl(masked, file, save_dir)
        print('>>> WRITING TO SHP???')
        masked = write_masked_shp(masked, file, shp_save_dir)
    else:
        print(f'>>> {file[-39:]} does not intersect so lets remove it')
        shutil.move(file, os.path.join(os.path.join(base_dir, tmp_dir), file[-39:]))
        return None
    return masked


iceshelf_dict = load_ice_shelf(iceshelf_filename, 'Amery')

if __name__ == '__main__':
    client = Client(n_workers=N_WORKERS)
    print(client)
    tasks = []
    start_time = timer.process_time()
    for file in data_list:
        mask = dask.delayed(filter_mask_write)(file)
        tasks.append(mask)
    results = dask.compute(*tasks)  # , traverse=False)
    print(f'total time taken : {(timer.process_time() - start_time) / 60} min')
