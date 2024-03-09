import numpy as np
import xarray as xr
from scipy import optimize
from scipy.special import jv, kv

from tools.dem_utils import sample_along_profile
from tools.utils import ll2xy, load_pkl

# define constants consistent with previous publications
g = 9.81  # m/s^2, gravity
rho_sw = 1028.  # kg/m^3, seawater density
rho_fw = 1000.  # kg/m^3, fresh water density
rho_i = 910.  # kg/m^3 ice density
nu = 0.3  # Poisson's ratio of ice

# define some reasonable bounds
# bounds = [lower, upper]
# Young's Modulus (Pa)
E_bounds = [5e6, 10e9]
# Ice shelf thickness (m) as bound by BedMachine v2 ice shelf thickness in the region
H_bounds = [865., 937.]
# Radius of supraglacial lake (m) as bound by measurements of the major and minor axes
Ld_bounds = [100., 800.]
# Depth of supraglacial lake (m)
d_bounds = [1, 50]
# collect them into a single input
BOUNDS = np.vstack([E_bounds, H_bounds, Ld_bounds, d_bounds]).T

E_0 = 50e6
H_0 = 868.
Ld_0 = 600.
d_0 = 20.


# define kelvin functions used as part of the analytical solution
def JV(x, order, a, b):
    return jv(order, x * np.exp(a * np.pi * 1j * b))


def KV(x, order, a, b):
    return kv(order, x * np.exp(a * np.pi * 1j * b))


def Ber(x):
    return np.real(JV(x, 0, 3, 1 / 4.))


def Bei(x):
    return np.imag(JV(x, 0, 3, 1 / 4.))


def Ker(x):
    return np.real(np.exp(-0 * np.pi * 1j / 2) * KV(x, 0, 1, 1 / 4.))


def Kei(x):
    return np.imag(np.exp(-0 * np.pi * 1j / 2) * KV(x, 0, 1, 1 / 4.))


# derivative of kelvin functions above
def Berp(x):
    arg = JV(x, 1, 3, 1 / 4.)
    return (np.real(arg) + np.imag(arg)) / np.sqrt(2)


def Beip(x):
    arg = JV(x, 1, 3, 1 / 4.)
    return (-np.real(arg) + np.imag(arg)) / np.sqrt(2)


def Kerp(x):
    pre = np.exp(-1 * np.pi * 1j / 2)
    arg = KV(x, 1, 1, 1 / 4.)
    return (np.real(pre * arg) + np.imag(pre * arg)) / np.sqrt(2)


def Keip(x):
    pre = np.exp(-1 * np.pi * 1j / 2)
    arg = KV(x, 1, 1, 1 / 4.)
    return (-np.real(pre * arg) + np.imag(pre * arg)) / np.sqrt(2)


def deflection(r, io, E, H, Ld, d):
    """
    The analytical solution to deflection of an ice shelf (elastic beam) loaded or de/unloaded by a meltwater lake (symmetrical cylinder load). See the following publications for more detail:
    MacAyeal & Sergienko (2013)
    Brotchie & Silvester (1969)
    Lambeck & Nakiboglu (1980)

    Parameters
    ---
    r : (n,1) np.ndarray
        Distance vector with r ~ 0 at the center in meters
    io : int
        Determined whether the ice shelf loading force.
        io = 0 : lake is gone, expect uplift
        io = 1 : lake is present, expect a depression
    E : float
        Young's Modulus (stiffness) in Pa
    H : float
        Ice shelf thickness in meters
    Ld : float
        Lake radius in meters
    d: float
        Lake depth in meters

    Returns
    ---
    u : (n,1) np.ndarray
        Deflection of the surface
    """
    # flexural rigidity
    D = E * H ** 3 / 12. / (1 - nu ** 2)
    # radius of flexural stiffness
    I = (D / rho_sw / g) ** 0.25

    # constants
    C1 = Ld * Kerp(Ld / I) / I
    C2 = -Ld * Keip(Ld / I) / I
    # primes of constants
    C1p = C2p = 0
    C3p = Ld * Berp(Ld / I) / I
    C4p = -Ld * Beip(Ld / I) / I

    # because the analytical solution is cylindrical and assumes axial symmetry,
    # we have to "remove" negative values of r
    r = np.abs(r)

    # preallocate solution
    u = np.zeros(r.size)
    # piecewise solution
    # u = u1 for r < Ld
    u1 = 1 + C1 * Ber(r / I) + C2 * Bei(r / I)
    # u = u2 for r >= Ld
    u2 = C1p * Ber(r / I) + C2p * Bei(r / I) + C3p * Ker(r / I) + C4p * Kei(r / I)
    # forcing
    F = (rho_fw * io - rho_i) * g * d
    u = - F * (u1 * (r < Ld) + u2 * (r >= Ld)) / g / rho_sw
    return u


def fitting_REMA_dh(is2_file, REMA_file,
                    f=deflection,
                    center_x=None, center_y=None,
                    beam="L", gt='gt1',
                    width=6e3,
                    initial_guess=(E_0, H_0, Ld_0, d_0), bounds=BOUNDS):
    """
    Fit REMA differences elevations to the analytical solution for elastic loading of an ice shelf by a meltwater lake.

    Parameters
    ----
    is2_file : string
        Path to the ICESat-2 file used for selecting the profile
    REMA_file : string
        Path to the REMA Lagrangian difference file for extracting the elevation differences along the profile
    center_x : float
        The center of the doline in the x-direction EPSG 3031 projection. Default = None (chooses middle cell of REMA DEM)
    center_y : float
        The center of the doline in the y-direction in EPSG 3031 projection. Default = None (chooses middle cell of REMA DEM)
    beam : string
        which
    width : float
    initial_guess : tuple/list
    bounds : np.ndarray

    Returns
    ----

    """
    ddm = xr.open_dataset(REMA_file)
    if center_x is None:
        center_x = np.median(ddm.x)
    if center_y is None:
        center_y = np.median(ddm.y)

    is2_data = load_pkl(is2_file)[gt]['masked']['Amery']

    bi = dict(L=0, R=1)[beam]
    # grab lat and lon of is2 track
    lat, lon = is2_data['latitude'][bi], is2_data['longitude'][bi]
    # remove the mask...
    ix = ~(np.isnan(lat) & np.isnan(lon))
    lat, lon = lat[ix], lon[ix]
    # convert to EPSG3031 distance
    x_is2, y_is2 = ll2xy(lon, lat)
    # grab IS2 ATl06 elevations
    h_is2 = is2_data['h_li'][bi][ix]

    # clean all of the above of only +/- width from the defined center of the doline
    ix = (((x_is2 - center_x >= -width) & (x_is2 - center_x <= width)) &
          (y_is2 - center_y >= -width) & (y_is2 - center_y <= width))
    x_is2, y_is2, h_is2 = x_is2[ix], y_is2[ix], h_is2[ix]
    lat, lon = lat[ix], lon[ix]

    # sample REMA dh along cleaned profiles
    dh_rema = sample_along_profile(ddm, 'dh', x_is2, y_is2)

    # now we have to clean the sampled profile
    ix = (~((np.isnan(dh_rema)) | (dh_rema < -100)) & (dh_rema < 40))
    x_is2, y_is2, h_is2 = x_is2[ix], y_is2[ix], h_is2[ix]
    lat, lon = lat[ix], lon[ix]
    dh_rema = dh_rema[ix]
    # demean the dh data
    dh_rema -= np.mean(dh_rema)
    # create the cylindrical coordinate; we use x because the y coordinate is a little too short in length scale
    r = x_is2 - center_x
    #     r = y_is2 - center_y
    # sometimes the REMA lagrangian differencing causes elevations that are unphysical on the edges, just due to the
    # process. here is how i'm automating them out
    ddhdr = np.gradient(dh_rema) / np.gradient(r)
    if np.any(abs(ddhdr[:3]) > (np.mean(ddhdr) + 3 * np.std(ddhdr))):
        i1 = 3
    else:
        i1 = 0
    if np.any(abs(ddhdr[-3:]) > (np.mean(ddhdr) + 3 * np.std(ddhdr))):
        i2 = -3
    else:
        i2 = -1
    ix = slice(i1, i2)
    r, dh_rema = r[ix], dh_rema[ix]
    lat, lon, x_is2, y_is2, h_is2 = lat[ix], lon[ix], x_is2[ix], y_is2[ix], h_is2[ix]
    # now create the data needed for the fitting to work properly
    # find the center index where r = 0
    center_ix = np.where(np.isclose(r, 0, rtol=0.1, atol=20))[0][0]
    # find where the elevation difference is highest from the right (or +r)
    hmax_R = np.where(dh_rema[center_ix:] == dh_rema[center_ix:].max())[0][0]
    # find where elevation difference is highest from the left (or -r)
    hmax_L = np.where(dh_rema[:center_ix] == dh_rema[:center_ix].max())[0][0]

    print(hmax_L, hmax_R)
    if hmax_R > hmax_L:
        hmax_L = center_ix + hmax_L
    elif hmax_R < hmax_L:
        hmax_R = center_ix + hmax_R

    # add a small buffer
    ix = slice(hmax_L + 3, hmax_R - 3)
    # this seems to capture the center of the off-center nature of some icesat2 tracks
    center_ix = int(0.5 * (hmax_R + hmax_L))
    #     print(center_ix)
    if r[center_ix] != 0:
        r = r - r[center_ix] + 1e-13

    # fitting data; since none of our tracks go over the direct uplift, we have to "mask" out the center depression
    R = r.copy()
    R[ix] = np.nan
    DH = dh_rema.copy()
    DH[ix] = np.nan

    bounds[:, 2] = [0.1e3, min([abs(r[hmax_L]), abs(r[hmax_R])]) - 20]
    if initial_guess[2] > bounds[1, 2] or initial_guess[2] < bounds[0, 2]:
        initial_guess[2] = 0.5 * (bounds[0, 2] + bounds[1, 2])

    C_C, _ = optimize.curve_fit(lambda x, E, H, Ld, d: f(x, 0, E, H, Ld, d),
                                R[~np.isnan(R)], DH[~np.isnan(DH)],
                                bounds=bounds,
                                p0=initial_guess,
                                maxfev=10_000,
                                # sigma=1 * np.ones(R[~np.isnan(R)].size), absolute_sigma=True,
                                )
    #     C_C = flex_fit(DH[~np.isnan(DH)], R[~np.isnan(R)], 0, list(initial_guess), bounds)
    down = np.zeros(r.size)
    # down[ix] += 2*dh_rema.min()
    down[ix] += (- 910) * 9.81 * C_C[-1] / 1028 / 9.81
    uplift = deflection(r, 0, *C_C)
    surface = uplift + down
    return x_is2, y_is2, lat, lon, h_is2, r, dh_rema, uplift, surface, C_C
