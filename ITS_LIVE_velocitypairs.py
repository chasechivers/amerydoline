import datetime
import json
import os

import numpy as np
import requests
import xarray as xr

from tools.utils import xy2ll, create_polydict_ptsarr

# bounding box in EPSG 4326, required bny ITS_LIVE API
# left, right, bottom, top = 69.3977595417875, 69.77300204362047, -72.09363441550481, -71.97497860824765
bottom, left, top, right = -72.2212, 68.3132, -71.8716, 70.8458

# dates
YEAR = 2015
start_date = f'{YEAR}-03-01'
end_date = f'{YEAR}-05-01'

#
save_dir = f'/data/MEaSUREs_ITS_LIVE/velocity_pairs/{YEAR}'

dt_fmt = '%Y-%m-%d'
min_interval = 1  # days
max_interval = 61  # days
base_url = "https://nsidc.org/apps/itslive-search/velocities/urls/"
params = {
    'bbox': '{0},{1},{2},{3}'.format(left, bottom, right, top),
    'start': start_date,
    'end': end_date,
    "percent_valid_pixels": 20,
    "min_interval": min_interval,
    "max_interval": max_interval,
    "version": 2  # version 1 requires an EDL access token header in the request
}

velocity_pairs = requests.get(base_url, params=params)
json_filename = os.path.join(save_dir, 'velocity_pairs_{0}-{1}_X_list.json'.format(start_date, end_date))
open(json_filename, 'wb').write(velocity_pairs.content)
file_list = json.load(open(json_filename, 'rb'))

dt_fmt2 = '%Y%m%d'
for img_url in file_list[:]:
    nc_file = img_url['url']
    file_name = nc_file.split('/')[-1]
    first_date, second_date = file_name.split('_')[3], file_name.split('_')[11]
    first_date = datetime.datetime.strptime(first_date, dt_fmt2)
    second_date = datetime.datetime.strptime(second_date, dt_fmt2)
    # if (datetime.datetime.strptime(start_date, dt_fmt) <= first_date) and \
    # if (second_date <= datetime.datetime.strptime(end_date, dt_fmt)):
    r = requests.get(nc_file)
    pair_savename = os.path.join(save_dir, file_name)
    print('saving ', pair_savename)
    open(pair_savename, 'wb').write(r.content)

########################################################################
# Clean files to only include those within specified bounding box
########################################################################

bb_filename = '/Volumes/RSRCHDATA/AmeryDolineProject/GIS/BoundingBoxes2.shp'
bbdict, bbpts = create_polydict_ptsarr(bb_filename)
dlshp = '/Volumes/RSRCHDATA/AmeryDolineProject/GIS/DolineOutlines.shp'
dldict, dlpts = create_polydict_ptsarr(dlshp)

extent = bbdict[0].bounds
left, bottom, right, top = extent
ll = xy2ll(left, bottom), xy2ll(left, top), xy2ll(right, top), xy2ll(right, bottom)
ll = np.array(ll)
lon_range = [ll[:, 0].min(), ll[:, 0].max()]
lat_range = [ll[:, 1].min(), ll[:, 1].max()]

extend_doline_region = 500
doline_extent = [np.inf, np.inf, -np.inf, -np.inf]

for k, v in dldict.items():
    bounds = v.bounds
    doline_extent[0] = bounds[0] if bounds[0] < doline_extent[0] else doline_extent[0]
    doline_extent[1] = bounds[1] if bounds[1] < doline_extent[1] else doline_extent[1]
    doline_extent[2] = bounds[2] if bounds[2] > doline_extent[2] else doline_extent[2]
    doline_extent[3] = bounds[3] if bounds[3] > doline_extent[3] else doline_extent[3]

dt_fmt = '%Y%m%d'

file_list = [f for f in os.listdir(save_dir) if '.nc' in f and f'{YEAR}' in f and '._' not in f]

for i, f in enumerate(file_list):
    print(i + 1)
    ds = xr.open_dataset(os.path.join(save_dir, f))
    ds = ds.isel(
        x=((ds.x >= doline_extent[0] - extend_doline_region) & (ds.x <= doline_extent[2] + extend_doline_region)),
        y=((ds.y >= doline_extent[1] - extend_doline_region) & (ds.y <= doline_extent[3] + extend_doline_region))
    )
    # not in bounding box, remove file
    if ds.y.values.size == 0 or ds.x.values.size == 0:
        print(f, ' remove')
        os.remove(os.path.join(save_dir, f))
        do_plot = False
    # in bounding box, keep file
    elif np.any(~np.isnan(ds.vx.values)):
        pass
    # probably not in bounding box, remove
    else:
        os.remove(os.path.join(save_dir, f))
