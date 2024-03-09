import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely

from utils import get_IS2_transformer

def plot_latlon_lines(ax, lat_range, lon_range,
                      extents,
                      transformer=None,
                      lon_step=0.15, lat_step=0.05,
                      line_style_dict=None, text_style_dict=None,
                      lon_text=True, lat_text=True,
                      ignore_lon_intersect=False, ignore_lat_intersect=False
                      ):
    """
    Function to plot latitude-longitude lines over images/etc whose main coordinates are not latitude-longitude, e.g. EPGS4326 lat-lon over EPSG3031 x,y

    Parameters
    ---
    ax : matplotlib.axes._subplots.AxesSubplot, e.g.
        Subplot of a figure
    lat_range : list, tuple
        Range of latitudes in plot (minimum, maximum)
    lon_range : list, tuple
        Range of longitudes in plot (minimum, maximum)
    extents : list, tuple
        Extent of plot in (x,y) coordinates (e.g. EPSG3031) in format (left, bottom, right, top)

    *optional*
    transformer : matplotlib.transforms.BboxTransformTo (default = None)
        Transforms coordinates to plot coordinates. If defualt = None, uses a transformer for EPSG4326 (lon, lat) to EPSG3031 (x,y).
        You could also use transformer = ax.transAxes,
    lon_step : float (default = 0.15)
        Step between longitude lines
    lat_step : float (default = 0.05)
        Step between latitude lines
    lon_text : bool (default = True)
        Whether to show the longitude line labels (e.g. 40 W) at the edges of the plot
    lat_text : bool (default = True)
        Whether to show the latitude line labels (e.g. 40 S) at the edges of the plot
    ignore_lon_intersect : bool (default = False)
    ignore_lat_intersect : bool (default = False)

    Returns
    ---
    ax : matplotlib.axes._subplots.AxesSubplot, e.g.
    """
    if transformer is None:
        transformer = get_IS2_transformer('S')
    if line_style_dict is None:
        line_style_dict = dict(c='k', ls='--', lw=0.3, alpha=1)
    if text_style_dict is None:
        text_style_dict = dict(color='grey', size=11)

    left, right, bottom, top = extents
    lats = np.arange(np.floor(lat_range[0]), np.ceil(lat_range[1]) + lat_step, lat_step)
    lons = np.arange(np.floor(lon_range[0]), np.ceil(lon_range[1]) + lon_step, lon_step)
    topline_xy = [[left, right], [top, top]]
    bottomline_xy = [[left, right], [bottom, bottom]]
    leftline_xy = [[left, left], [bottom, top]]
    rightline_xy = [[right, right], [bottom, top]]

    line_2 = shapely.geometry.LineString(np.c_[leftline_xy[0], leftline_xy[-1]])
    line_3 = shapely.geometry.LineString(np.c_[bottomline_xy[0], bottomline_xy[1]])
    for lon_i in lons:
        LON_I, LAT_I = lon_i * np.ones_like(lats), lats
        X_I, Y_I = transformer.transform(LON_I, LAT_I)
        line_1 = shapely.geometry.LineString(np.c_[X_I, Y_I])

        intersects = line_1.intersects(line_2) or line_1.intersects(line_3)
        if (not ignore_lon_intersect and intersects):
            ax.plot(X_I, Y_I, **line_style_dict)
            if line_1.intersects(line_2):
                intersections = line_1.intersection(line_2)
            else:
                intersections = line_1.intersection(line_3)
            x_int, y_int = intersections.x, intersections.y
            if lon_text and x_int <= left:
                degree_label = f'{lon_i:0.4g}°E'
                ax.text(x_int, y_int, degree_label,
                        verticalalignment='center', horizontalalignment='right', rotation='vertical',
                        **text_style_dict)

    line_2 = shapely.geometry.LineString(np.c_[topline_xy[0], topline_xy[1]])
    for lat_i in lats:
        LON_I, LAT_I = lons, lat_i * np.ones_like(lons)
        X_I, Y_I = transformer.transform(LON_I, LAT_I)
        line_1 = shapely.geometry.LineString(np.c_[X_I, Y_I])

        intersects = line_1.intersects(line_2)
        if (not ignore_lat_intersect and intersects):
            intersections = line_1.intersection(line_2)
            x_int, y_int = intersections.x, intersections.y
            ax.plot(X_I, Y_I, **line_style_dict)

            if lat_text:
                degree_label = f'{abs(lat_i):0.4g}°{"S" if lat_i < 0 else "N"}'
                ax.text(x_int, y_int, degree_label,
                        verticalalignment='bottom', horizontalalignment='center',
                        **text_style_dict)

    return ax


def shifted_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero.

    Taken from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Parameters
    -----
    cmap :
        The matplotlib colormap to be altered
    start :
        Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
    midpoint :
        The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
    stop :
        Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.

    Returns
    ----
    newcmap : matplotlib.colors.ListedColormap
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
