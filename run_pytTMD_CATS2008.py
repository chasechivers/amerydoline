import pyTMD
import os
import numpy as np
from tools.utils import xy2ll
import xarray as xr
import datetime

save_dir = 'data'
date1 = datetime.datetime(2014, 3, 1)
date2 = datetime.datetime(2015, 5, 1)

left, right = 1590000.674942, 2023427.3908214455
bottom, top = 601687.415923, 845588.645856

xlimits = [left, right]
ylimits = [bottom, top]
spacing = [1e3, -1e3]
x = np.arange(xlimits[0], xlimits[1] + spacing[0], spacing[0])
y = np.arange(ylimits[1], ylimits[0] + spacing[1], spacing[1])
xgrid, ygrid = np.meshgrid(x, y)
ny, nx = xgrid.shape
lon, lat = xy2ll(xgrid.flatten(), ygrid.flatten())

path_to_def_file = 'data/model_CATS2008.def'
md = pyTMD.io.model().from_file(path_to_def_file).elevation('CATS2008')
amplitude, phase, D, constituents = pyTMD.io.OTIS.extract_constants(lon, lat, md.grid_file, md.model_file,
                                                                    grid=md.format)

cph = -1j * phase * np.pi / 180.0
hc = amplitude * np.exp(cph)
NDAYS = (date2 - date1).days
time = np.array([date1 + datetime.timedelta(days=d) for d in range(NDAYS)])
max_tide = np.ma.zeros((ny, nx, NDAYS))
min_tide = max_tide.copy()
diff_tide = max_tide.copy()
# count = 0
for day in range(NDAYS):
    date = date1 + datetime.timedelta(days=day)
    print(date)
    tide_time = pyTMD.time.convert_calendar_dates(date.year, date.month, date.day, hour=np.arange(24))
    DELTAT = np.zeros_like(tide_time)
    tmp = []
    for hour in range(24):
        print('\t', hour + 1)
        TIDE = pyTMD.predict.map(tide_time[hour], hc, constituents, deltat=DELTAT[hour],
                                 corrections=md.format)
        MINOR = pyTMD.predict.infer_minor(tide_time[hour], hc, constituents, deltat=DELTAT[hour],
                                          corrections=md.format)
        tmp.append(np.reshape((TIDE + MINOR), (ny, nx)))
    max_tide[:, :, day] = np.max(tmp, axis=0)
    min_tide[:, :, day] = np.min(tmp, axis=0)
    diff_tide[:, :, day] = max_tide[:, :, day] - min_tide[:, :, day]

# save to xarray
ds = xr.Dataset(data_vars=dict(
    daymax=(['y', 'x', 'time'], max_tide),
    daymin=(['y', 'x', 'time'], min_tide),
    dayrange=(['y', 'x', 'time'], diff_tide)
),
    coords=dict(x=x,
                y=y,
                time=time,
                lon=(['y', 'x'], lon.reshape(ny, nx)),
                lat=(['y', 'x'], lat.reshape(ny, nx))
                )
)

date2string = lambda x: '{0}-{1}-{2}'.format(x.year, f'{x.month}'.zfill(2), f'{x.day}'.zfill(2))
ds.to_netcdf(os.path.join(save_dir, 'AmeryIS_tides_CATS2008_{0}_{1}.nc'.format(date2string(date1), date2string(date2))))