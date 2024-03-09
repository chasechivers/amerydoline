import datetime
import itertools
import json
import os
import re

import numpy as np
import rasterio as rio
import rasterio.mask as riomask

# LandSat mission bands and their info
# BAND: {PLAIN LANGUAGE DESCRIPTION, SPATIAL RESOLUTION, WAVELENGTH RANGE}
LANDSAT1_BAND_INFO = {
    'B4': {'desc.': 'Green', 'resolution (m)': 60, 'wavelength (µm)': '0.5-0.6'},
    'B5': {'desc.': 'Red', 'resolution (m)': 60, 'wavelength (µm)': '0.6-0.7'},
    'B6': {'desc.': 'NIR', 'resolution (m)': 60, 'wavelength (µm)': '0.7-0.8'},
    'B7': {'desc.': 'NIR', 'resolution (m)': 60, 'wavelength (µm)': '0.8-1.1'}
}
LANDSAT4_BAND_INFO = {
    'B1': {'desc.': 'Blue', 'resolution (m)': 30, 'wavelength (µm)': '0.45-0.52'},
    'B2': {'desc.': 'Green', 'resolution (m)': 30, 'wavelength (µm)': '0.52-0.60'},
    'B3': {'desc.': 'Red', 'resolution (m)': 30, 'wavelength (µm)': '0.63-0.69'},
    'B4': {'desc.': 'NIR', 'resolution (m)': 30, 'wavelength (µm)': '0.76-0.90'},
    'B5': {'desc.': 'SWIR1', 'resolution (m)': 30, 'wavelength (µm)': '1.55-1.75'},
    'B6': {'desc.': 'Thermal', 'resolution (m)': 30, 'wavelength (µm)': '10.40-12.50'},
    'B7': {'desc.': 'SWIR2', 'resolution (m)': 30, 'wavelength (µm)': '2.08-2.35'}
}
LANDSAT4_BAND_INFO = {
    'B1': {'desc.': 'Blue', 'resolution (m)': 30, 'wavelength (µm)': '0.45-0.52'},
    'B2': {'desc.': 'Green', 'resolution (m)': 30, 'wavelength (µm)': '0.52-0.60'},
    'B3': {'desc.': 'Red', 'resolution (m)': 30, 'wavelength (µm)': '0.63-0.69'},
    'B4': {'desc.': 'NIR', 'resolution (m)': 30, 'wavelength (µm)': '0.76-0.90'},
    'B5': {'desc.': 'SWIR1', 'resolution (m)': 30, 'wavelength (µm)': '1.55-1.75'},
    'B6': {'desc.': 'Thermal', 'resolution (m)': 30, 'wavelength (µm)': '10.40-12.50'},
    'B7': {'desc.': 'SWIR2', 'resolution (m)': 30, 'wavelength (µm)': '2.08-2.35'}
}
LANDSAT7_BAND_INFO = {
    'B1': {'desc.': 'Blue', 'resolution (m)': 30, 'wavelength (µm)': '0.45-0.52'},
    'B2': {'desc.': 'Green', 'resolution (m)': 30, 'wavelength (µm)': '0.52-0.60'},
    'B3': {'desc.': 'Red', 'resolution (m)': 30, 'wavelength (µm)': '0.63-0.69'},
    'B4': {'desc.': 'NIR', 'resolution (m)': 30, 'wavelength (µm)': '0.78-0.90'},
    'B5': {'desc.': 'SWIR1', 'resolution (m)': 30, 'wavelength (µm)': '1.55-1.75'},
    'B6': {'desc.': 'Thermal', 'resolution (m)': 30, 'wavelength (µm)': '10.40-12.50'},
    'B7': {'desc.': 'SWIR2', 'resolution (m)': 30, 'wavelength (µm)': '2.09-2.35'},
    'B8': {'desc.': 'Pan', 'resolution (m)': 15, 'wavelength (µm)': '0.52-0.90'},
}
LANDSAT8_BAND_INFO = {
    'B1': {'desc.': 'Coastal aerosol', 'resolution (m)': 30, 'wavelength (µm)': '0.43-0.45'},
    'B2': {'desc.': 'Blue', 'resolution (m)': 30, 'wavelength (µm)': '0.43-0.45'},
    'B3': {'desc.': 'Green', 'resolution (m)': 30, 'wavelength (µm)': '0.45-0.51'},
    'B4': {'desc.': 'Red', 'resolution (m)': 30, 'wavelength (µm)': '0.64-0.67'},
    'B5': {'desc.': 'NIR', 'resolution (m)': 30, 'wavelength (µm)': '0.85-0.88'},
    'B6': {'desc.': 'SWIR1', 'resolution (m)': 30, 'wavelength (µm)': '1.57-1.65'},
    'B7': {'desc.': 'SWIR2', 'resolution (m)': 30, 'wavelength (µm)': '2.11-2.29'},
    'B8': {'desc.': 'Pan', 'resolution (m)': 15, 'wavelength (µm)': '0.5-0.68'},
    'B9': {'desc.': 'Cirrus', 'resolution (m)': 30, 'wavelength (µm)': '1.36-1.38'},
    'B10': {'desc.': 'TIRS1', 'resolution (m)': 100, 'wavelength (µm)': '10.6-11.19'},
    'B11': {'desc.': 'TIRS2', 'resolution (m)': 100, 'wavelength (µm)': '11.5-12.51'}
}


def normalize_band(a, MIN=None, MAX=None):
    """
    Normalize a raster to its minimum and maximum values.

    Parameters
    ---
    a : np.ndarray
        Array of values
    MIN : float (default = None)
        Minimum value to normalize to. Defaults to minimum value of a
    MAX : float (defualt = None)
        Maximum value to normalize to. Defaults to maximum value of a

    Returns
    ---
    n : np.ndarray
        Normalized array
    """
    if MIN is None:
        MIN = np.nanmin(a)
    if MAX is None:
        MAX = np.nanmax(a)
    n = (a - MIN) / (MAX - MIN)
    return n


def TOA_reflectance(Q_cal, M_rho, A_rho, theta_SE):
    """
    Calculate the top-of-atmosphere (TOA) reflectance for LandSat8-9 image data with a correction for the sun angle. Sourced from
    https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

    Parameters
    ---
    Q_cal : np.ndarray, float
        Quantized and calibrated standard product pixel values (DN)
    M_rho : float
        Band-specific multiplicative rescaling factor from the metadata (REFLECTANCE_MULT_BAND_x, where x is the band number)
    A_rho : float
        Band-specific additive rescaling factor from the metadata (REFLECTANCE_ADD_BAND_x, where x is the band number)
    theta_SE : float
        Local sun elevation angle in degrees. The scene center sun elevation angle in degrees is provided in the metadata (SUN_ELEVATION).

    Returns
    ---
    rho_lambda : np.ndarray, float
        TOA planetary reflectance
    """
    # theta_SE = np.deg2rad(theta_SE)
    return (M_rho * Q_cal + A_rho) / np.sin(theta_SE)


def TOA_radiance(Q_cal, M_L, A_L):
    """
    Calculate the top-of-atmosphere (TOA) radiance for LandSat8 image data. Sourced from https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

    Parameters
    ---
    Q_cal : np.ndarray, float
        Quantized and calibrated standard product pixel values (DN)
    M_L : float
        Band-specific multiplicative rescaling factor from the metadata (RADIANCE_MULT_BAND_x, where x is the band number)
    A_L : float
        Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x, where x is the band number)

    Returns
    ---
    L_lambda : np.ndarray, float
        TOA spectral radiance (W/m^2/srad/micrometer)

    """
    return M_L * Q_cal + A_L


def TOA_brightness_temperature(L_lambda, K2, K1):
    """
    Calculate the top-of-atmosphere (TOA) brightness temperature for LandSat8 image data. Sourced from https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

    Parameters
    ---
    L_lambda : np.ndarray, float
        TOA spectral radiance (W/m^2/srad/micrometer)
    K1 : float
        Band-specific thermal conversion constant from the metadata (K1_CONSTANT_BAND_x, where x is the thermal band number)
    K2 : float
        Band-specific thermal conversion constant from the metadata (K2_CONSTANT_BAND_x, where x is the thermal band number)

    Returns
    ---
    T : np.ndarray, float
        Top of atmosphere brightness temperature (K)
    """
    return K2 / np.log(K1 / L_lambda + 1)


def moussavi_lakes_mask(bands, TIRS1,
                        NDWI_threshold=0.19,
                        B3B4_threshold=0.07,
                        B2B3_threshold=0.11):
    """
    Create a binary (0, 1) mask of where supgraglacial lakes and channels are present on the ice following the methodology of Moussavi et al. 2020.

    Thresholds help determine where masks should be and can be adjusted, but the defaults are given by Moussavi et al. 2020 and Trusel et al. 2022

    Parameters
    ---
    bands : dictionary
        Dictionary of relevant bands that have been at least top-of-atmosphere (TOA) reflectance corrected, bonus points for upscaling and pan-sharpening.
        Dictionary should have structure, e.g., if we want the blue and red bands from LandsSat8-9
            blue = bands['B2']
            red = bands['B4']
    TIRS1 : array
        Brightness temperature of TIRS1 band ('B10')
    NDWI_threshold : float
        Threshold value for the Normalized Difference Water Index (Blue Red). Default = 0.18
    B3B4_threshold : tuple, list
        Threshold bounds for the B3 - B4 (Green - Red) difference. Default = 0.07
    B2B3_threshold : float
        Threshold for the B2 - B3 (Blue - Green) difference. Default = 0.04

    Returns
    ---
    lake_mask : array

    """
    NDWI = (bands['B2'] - bands['B4']) / (bands['B2'] + bands['B4'])
    B3B4 = bands['B3'] - bands['B4']
    B2B3 = bands['B2'] - bands['B3']
    lake_mask = ((NDWI > NDWI_threshold) & (B3B4 > B3B4_threshold) & (B2B3 > B2B3_threshold)).astype(np.int64)

    # cloud mask
    NDSI = (bands['B3'] - bands['B6']) / (bands['B3'] + bands['B6'])
    cloudmask = ((NDSI < 0.8) & (bands['B6'] > 0.1)).astype(np.int64)

    # rock mask
    rockmask = ((TIRS1 / bands['B2'] > 650) & (bands['B2'] < 0.35)).astype(np.int64)

    lake_mask = lake_mask & (~cloudmask) & (~rockmask)
    return lake_mask


def corr_lakes_mask(bands,
                    NDWIGN_threshold=0.16,
                    NDWIBR_threshold=0.18,
                    B3B4_bounds=(0.08, 0.4),
                    B2B3_threshold=0.04,
                    NDSI_max_threshold=0.8,
                    B2_bounds=(0.6, 0.95),
                    B6_threshold=0.1):
    """
    Create a binary mask of where supraglacial lakes and channels are present on the ice shelf following the methodology of Corr et al. 2022 (see Figure 3 in the text).

    Thresholds help determine where masks should be and can be adjusted, but the defaults are given by Corr et al. 2022

    Parameters
    ---
    bands : dictionary
        Dictionary of relevant bands that have been at least top-of-atmosphere (TOA) reflectance corrected, bonus points for upscaling and pan-sharpening.
        Dictionary should have structure, e.g., if we want the blue and red bands from LandsSat8-9
            blue = bands['B2']
            red = bands['B4']
    NDWIGN_threshold : float
        Threshold value for the Normalized Difference Water Index (Green NIR). Default = 0.16
    NDWIBR_threshold : float
        Threshold value for the Normalized Difference Water Index (Blue Red). Default = 0.18
    B3B4_bounds : tuple, list
        Threshold bounds for the B3 - B4 difference. Default = (> 0.08, < 0.4)
    B2B3_threshold : float
        Threshold for the B2 - B3 difference. Default = 0.04
    NDSI_max_threshold : float
        Maximum threshold value for the Normalized Difference Snow Index. Default = 0.8 (i.e. NDSI < NDSI_max_threshold)
    B2_bounds : tuple, list
        Threshold bounds for the B2 band (blue). Default = (> 0.6, < 0.95)
    B6_threshold : float
        Threshold for the B6 band (SWIR). Default = 0.1

    Returns
    ---
    water_mask : np.ndarray, np.ma.array
        Binary mask of all water present in scene
    clouds_mask : np.ndarray, np.ma.array
        Binary mask of clouds
    shadows_mask : np.ndarray, np.ma.array
        Binary mask of snow and cloud shadow
    lake_mask : np.ndarray, np.ma.array
        Binary mask of supraglacial lakes/channels
    """
    assert ('B2' in bands) & ('B4' in bands) & ('B3' in bands) & ('B5' in bands) & ('B6' in bands), \
        'bands B2, B3, B4, B5, and B6 all need to be in the bands dictionary object'
    blue, green, red, nir, swir = bands['B2'], bands['B3'], bands['B4'], bands['B5'], bands['B6']
    # mask = blue.mask
    # preallocate array for each mask,
    NDWIBR = (blue - red) / (blue + red)
    NDWIGN = (green - nir) / (green + nir)

    B3B4 = (green - red)
    B2B3 = (blue - green)

    NDSI = np.ma.array(np.where((green + swir) == 0, 0, (green - swir) / (green + swir)),
                       mask=green.mask)

    B2 = blue  # / np.max(blue)

    B6 = swir  # / np.max(swir)

    water_mask = (NDWIGN > NDWIGN_threshold) & (NDWIBR > NDWIBR_threshold)
    clouds_mask = (NDSI < NDSI_max_threshold) & (B6 > B6_threshold) \
                  & ((B2 > B2_bounds[0]) & (B2 < B2_bounds[1]))
    shadows_mask = ((B3B4 > B3B4_bounds[0]) & (B3B4 < B3B4_bounds[1])) & (B2B3 > B2B3_threshold)
    lake_mask = water_mask & ~(shadows_mask | clouds_mask)

    return water_mask, clouds_mask, shadows_mask, lake_mask


class easy_raster:
    """
    Class object to load and keep the raster data and relevant raster information in a single object.

    Bands are stored in a dictionary (.bands) indexed by their name (i.e. 'B1', 'B2', etc.)
    """
    def __init__(self, directory, bands='all', mask=None, always_TOA=False, pan_band='B8'):
        """
        Initiate class object

        Parameters
        ---
        directory : str
            Path to where all Landsat geotiffs are stored
        bands : list, tuple, or str (default = 'all')
            List/tuple of bands to load. Default behavior will load all available bands as determined by the info dicts in the landsat_utils.py file
        mask : rasterio.mask object
            Spatial mask
        always_TOA : bool
            Whether to always use Top-of-Atmosphere Reflectance corrected bands in any calculations
        pan_band : str (default = 'B8' [from Landsat 8-9])
            Name of panchromatic band. Default assumes Landsat 8-9 ('B8')
        """
        if bands == 'all':
            bands = [f'{b[0]}{b[1]}' for b in itertools.product(['B'], range(1, 11 + 1))]
        self.band_names = bands
        self.mask = mask
        self._dir = directory
        files = os.listdir(directory)
        self.band_files = {b: [f for f in files if b in f and '._' not in f and '.aux' not in f][0] for b in bands}
        self.MTL_file = \
        [os.path.join(directory, f) for f in files if ('MTL.txt' in f or 'MTL.json' in f) and '._' not in f][0]
        self.bands = {b: [] for b in bands}
        self.masked = {b: [] for b in bands}
        for band_name, file_name in self.band_files.items():
            mosaic = rio.open(os.path.join(self._dir, file_name))
            if band_name == bands[0]:
                self.crs = mosaic.crs
                self.width = mosaic.width
                self.height = mosaic.height
                self.transform = mosaic.transform
                self.img_bounds = mosaic.bounds
            elif band_name == pan_band:
                self.pan_transform = mosaic.transform
                self.pan_crs = mosaic.crs
                self.pan_width = mosaic.width
                self.pan_height = mosaic.height
                self.pan_transform = mosaic.transform
                self.pan_img_bounds = mosaic.bounds
            else:
                pass
            self.bands[band_name] = mosaic.read()[0]
            if mask is None:
                pass
            else:
                try:
                    masked, _ = riomask.mask(mosaic,
                                             [mask],
                                             crop=True,
                                             filled=False
                                             )
                    self.masked[band_name] = masked[0]
                except:
                    raise Exception('Please pass a shapely object')
        print([f for f in files if ('{0}.TIF'.format(bands[0]) in f or '{0}.tif'.format(bands[0]) in f)])
        self._parse_info([f for f in files if
                          ('{0}.TIF'.format(bands[0]) in f or '{0}.tif'.format(bands[0]) in f) and '.aux' not in f][0])

        self._scrape_TOA_coeff()
        self.always_TOA = always_TOA
        if always_TOA:
            self.calculate_TOA()

    def _parse_info(self, file_name):
        """
        Parses information from the file_name
        """
        # file = self.band_files[k][-47:]
        rx = re.compile(
            r'L(\w{1})(\d{2})_L1(\w{2})_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{'
            r'2})_(\d{2})_T(\d{1})_B(\d{1}).(\w{3})'
        )
        _, LSN, _, PATH, ROW, Y1, M1, D1, Y2, M2, D2, N1, N2, BN, FT = rx.findall(file_name).pop()
        self.date = datetime.datetime(int(Y1), int(M1), int(D1))
        self.path = PATH
        self.row = ROW

    def create_xy(self, transform=None, height=None, width=None):
        """
        Create 2D x,y arrays for rasters if needed.
        """
        if transform is None:
            transform = self.transform
        if height is None:
            height = self.height
        if width is None:
            width = self.width

        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rio.transform.xy(transform, rows, cols)
        X = np.array(xs)
        Y = np.array(ys)
        x = X[0, :]
        y = Y[:, 0]
        return x, y

    def _scrape_TOA_coeff(self):
        """
        Scrapes Top-of-Atmosphere Reflectance coefficients from MTL (json or txt) files. Not all Landsat missions were implemented as only a few were necessary for analysis
        """
        RADIANCE_TERMS = ['RADIANCE_MULT',
                          'RADIANCE_ADD',
                          'REFLECTANCE_MULT',
                          'REFLECTANCE_ADD',
                          'K1_CONSTANT',
                          'K2_CONSTANT']

        OTHER_TERMS = ['SUN_ELEVATION']

        self.TOA_COEFF = {
            'bands': {b: {coeff: np.nan for coeff in RADIANCE_TERMS} for b in
                      self.band_names}
        }
        for term in OTHER_TERMS:
            self.TOA_COEFF[term] = np.nan

        # dictionary to store TOA reflectance values
        self.TOA = {b: np.nan for b in self.band_names}

        if '.json' in self.MTL_file:
            mtlfile = open(self.MTL_file, 'r')
            data = json.load(mtlfile)  # ['LANDSAT_METADATA_FILE']
            data = data['LANDSAT_METADATA_FILE']
            RAD_PARENT_KEY = 'LEVEL1_RADIOMETRIC_RESCALING'
            THERMAL_PARENT_KEY = 'LEVEL1_THERMAL_CONSTANTS'
            OTHER_PARENT_KEY = 'IMAGE_ATTRIBUTES'

            for band_coeff, value in data[RAD_PARENT_KEY].items():
                sp = band_coeff.split('_')
                band = '{0}{1}'.format('B', int(sp[-1]))
                coeff = '{0}_{1}'.format(sp[0], sp[1])
                if band in self.band_names:
                    self.TOA_COEFF['bands'][band][coeff] = float(value)
            if THERMAL_PARENT_KEY in data:
                try:
                    for band_coeff, value in data[THERMAL_PARENT_KEY].items():
                        sp = band_coeff.split('_')
                        coeff = '{0}_{1}'.format(sp[0], sp[1])
                        band = '{0}{1}'.format('B', int(sp[-1]))
                        if band in self.band_names:
                            self.TOA_COEFF['bands'][band][coeff] = float(value)
                except:
                    print('This is probably not implemented yet')

            for coeff, value in data[OTHER_PARENT_KEY].items():
                if coeff in OTHER_TERMS:
                    self.TOA_COEFF[coeff] = float(value)

        # json is so much easier, but apparently landsat9 level1 products only come with a text file :\
        elif '.txt' in self.MTL_file:
            mtlfile = open(self.MTL_file, 'r')
            lines = mtlfile.readlines()

            for bst in RADIANCE_TERMS:
                for line in lines:
                    if bst in line:
                        split1 = line.split('=')
                        value = float(split1[-1])
                        split2 = split1[0].split('_')
                        bandnum = int(split2[-1])
                        key = f'B{bandnum}'
                        if key in self.band_names:
                            self.TOA_COEFF['bands'][key][bst] = value

            for sst in OTHER_TERMS:
                for line in lines:
                    if sst in line:
                        value = float(line.split('=')[-1])
                        self.TOA_COEFF[sst] = value

    def calculate_TOA(self, bands=None):
        """
        Calculate Top-of-Atmosphere Reflectance Correction for a raster.
        """
        for band in self.band_names:
            M_rho = self.TOA_COEFF['bands'][band]['REFLECTANCE_MULT']
            A_rho = self.TOA_COEFF['bands'][band]['REFLECTANCE_ADD']
            theta_SE = np.deg2rad(self.TOA_COEFF['SUN_ELEVATION'])
            if bands is None:
                Q_cal = self.masked[band] if self.mask is not None else self.bands[band]
            else:
                Q_cal = bands[band]
            if ~np.isnan(M_rho):
                self.TOA[band] = TOA_reflectance(Q_cal, M_rho, A_rho, theta_SE)
            else:
                self.TOA[band] = Q_cal

        return self.TOA

    def subtract_bands(self, band1, band2):
        """generalized method for adding bands"""
        if self.always_TOA:
            return self.TOA[band1] - self.TOA[band2]
        else:
            d = self.masked if self.mask is not None else self.bands
            return d[band1] - d[band2]

    def add_bands(self, band1, band2):
        """generalized method for subtracting bands"""
        if self.always_TOA:
            return self.TOA[band1] + self.TOA[band2]
        else:
            d = self.masked if self.mask is not None else self.bands
            return d[band1] + d[band2]

    def calculate_brightness_temperature(self, bands=None):
        """
        Calculates brightness temperature for correction/lake masking
        """
        if bands is None:
            bands = self.masked if self.mask is not None else self.bands
        TIRS = bands['B10']
        # convert to radiance
        radiance = TOA_radiance(TIRS,
                                self.TOA_COEFF['bands']['B10']['RADIANCE_MULT'],
                                self.TOA_COEFF['bands']['B10']['RADIANCE_ADD'])
        # convert to brightness temperature
        K1 = self.TOA_COEFF['bands']['B10']['K1_CONSTANT']
        K2 = self.TOA_COEFF['bands']['B10']['K2_CONSTANT']
        brightness_temperature = TOA_brightness_temperature(radiance, K2, K1)
        return brightness_temperature

    def pan_sharpen(self, method='IHS', weight=0.2, TOA_correction=True,
                    red='B4', green='B3', blue='B2', pan='B8', SWIR='B6'):
        """
        Pan sharpen and up-sample all bands with a given method

        Parameters
        ---
        method : string (default = 'IHS')
            Method used to pan-sharpen RGB (and NIR if using the intensity hue saturation (IHS) method). Default is 'IHS' method, but I prefer 'brovey' for most applications
            Options:
                'IHS' - Intensity Hue Saturation method (Rahmani et al., 2010)
                S = 1/4 * (Blue + Red + Green + Near Infrared)
                Band = Band + (Pan - S)

                'Brovey' - Brovey method ()
                Ratio = (2 + weight) * Pan / (Red + Green + weight * Blue)

        weight : float (optional)
            Weighted factor for blue band in Brovey method. Default = True
        TOA_correction : bool (optional)
            Whether to apply top-of-atmosphere correction
        red : str
        green : str
        blue : str
        pan : str
        SWIR : str

        Returns
        ---
        pans_up : dict
            Pansharpened and up-sampled bands
        """
        pans_up = {b: None for b in self.band_names}
        pan_band = pan
        if self.mask is not None:
            d = self.masked
            mask = d[pan_band].mask
        else:
            d = self.bands

        pan = d[pan_band]
        pans_up[pan_band] = pan.copy()
        band_names = [b for b in self.band_names if pan_band not in b]

        # up-sample all the bands to the pan resolution
        for band_name in band_names:
            band = d[band_name]
            up_band = np.empty(pan.shape, dtype=band.dtype)
            rio.warp.reproject(band, up_band,
                               src_transform=self.transform,
                               src_crs=self.crs,
                               dst_transform=self.pan_transform,
                               dst_crs=self.crs,
                               resampling=rio.enums.Resampling.bilinear
                               )
            if self.mask is not None:
                pans_up[band_name] = np.ma.array(up_band.copy(),
                                                 mask=mask.copy())
            else:
                pans_up[band_name] = up_band.copy()

        # pan-sharpen the relevant bands
        with np.errstate(divide='ignore', invalid='ignore'):
            if method.lower() == 'ihs':
                pansharpen_bands = [blue, green, red, SWIR]
                S = (1 / 4.) * (pans_up[blue] + pans_up[green] + pans_up[red] + pans_up[SWIR])
                # print(f'{pan.dtype=},{S.dtype=}')
                for band_name in pansharpen_bands:
                    # print(f'   {pans_up[band_name].dtype=}')
                    pans_up[band_name] = pans_up[band_name] + (pan - S)

            elif method.lower() == 'brovey':
                ratio = pan / ((pans_up[red] + pans_up[green] + pans_up[blue] * weight) / (2 + weight))
                rgb = np.array([pans_up[red], pans_up[green], pans_up[blue]])
                rgb = np.clip(ratio * rgb, 0, np.iinfo(pan.dtype).max)
                # print(f'{rgb.shape=}')
                pans_up[red] = rgb[0]
                pans_up[green] = rgb[1]
                pans_up[blue] = rgb[2]

        if TOA_correction:
            pans_up = self.calculate_TOA(pans_up)

        return pans_up

    def lake_mask(self):
        """
        "Find" and mask lakes using the method of Moussavi et al. 2020. See moussavi_lakes_mask function for more detail.
        """
        TIRS1 = self.calculate_brightness_temperature(bands=self.TOA)
        return moussavi_lakes_mask(self.TOA, TIRS1)

    def to_RGB(self, bands=None, red='B4', green='B3', blue='B2'):
        """
        Create an (ny,nx,3) array for plotting RGB images in matplotlib

        Parameters
        ---
        bands : dict (default = default bands used)
            Dictionary of bands (masked, TOA corrected, etc.) with keys related to their band names ('B1','B2',...)
        red : str (default = 'B4')
            Red band name. Default is B4 for Landsat 8-9
        green : str (default = 'B3')
            Green band name. Default is B4 for Landsat 8-9
        blue : str (default = 'B2')
            Blue band name. Default is B4 for Landsat 8-9

        Returns
        ---
        RGB : (ny,nx,3) np.ndarray
        """
        if bands is None:
            d = self.masked if self.mask is not None else self.bands
        else:
            d = bands

        R = d[red]
        G = d[green]
        B = d[blue]
        if self.mask is not None:
            mask = np.empty((R.shape[0], R.shape[1], 3), dtype=R.mask.dtype)
            mask[:, :, 0] = R.mask
            mask[:, :, 1] = R.mask
            mask[:, :, 2] = R.mask
            RGB = np.ma.array(np.empty((R.shape[0], R.shape[1], 3), dtype=R.dtype),
                              mask=mask)

        else:
            RGB = np.zeros((R.shape[0], R.shape[1], 3))
        RGB[:, :, 0] = R / np.nanmax(R)  # (R - np.nanmin(R)) / (np.nanmax(R) - np.nanmin(R))
        RGB[:, :, 1] = G / np.nanmax(G)  # (G - np.nanmin(G)) / (np.nanmax(G) - np.nanmin(G))
        RGB[:, :, 2] = B / np.nanmax(B)  # (B - np.nanmin(B)) / (np.nanmax(B) - np.nanmin(B))
        return RGB

    @property
    def NDWI_BR(self):
        return (self.blue - self.red) / (self.blue + self.red)

    @property
    def NDWI_GN(self):
        return (self.green - self.NIR) / (self.green + self.NIR)

    @property
    def blue(self):
        if self.always_TOA:
            return self.TOA['B2']
        else:
            return self.masked['B2'] if self.mask is not None else self.bands['B2']

    @property
    def red(self):
        if self.always_TOA:
            return self.TOA['B4']
        else:
            return self.masked['B4'] if self.mask is not None else self.bands['B4']

    @property
    def green(self):
        if self.always_TOA:
            return self.TOA['B3']
        else:
            return self.masked['B3'] if self.mask is not None else self.bands['B3']

    @property
    def pan(self):
        if self.always_TOA:
            return self.TOA['B8']
        else:
            return self.masked['B8'] if self.mask is not None else self.bands['B8']

    @property
    def NIR(self):
        if self.always_TOA:
            return self.TOA['B5']
        else:
            return self.masked['B5'] if self.mask is not None else self.bands['B5']
