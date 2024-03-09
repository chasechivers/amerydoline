#!/usr/bin/env python
# ----------------------------------------------------------------------------
# NSIDC Data Download Script
#
# Copyright (c) 2022 Regents of the University of Colorado
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# Tested in Python 2.7 and Python 3.4, 3.6, 3.7, 3.8, 3.9
#
# To run the script at a Linux, macOS, or Cygwin command-line terminal:
#   $ python nsidc-data-download.py
#
# On Windows, open Start menu -> Run and type cmd. Then type:
#     python nsidc-data-download.py
#
# The script will first search Earthdata for all matching files.
# You will then be prompted for your Earthdata username/password
# and the script will download the matching files.
#
# If you wish, you may store your Earthdata username/password in a .netrc
# file in your $HOME directory and the script will automatically attempt to
# read this file. The .netrc file should have the following format:
#    machine urs.earthdata.nasa.gov login MYUSERNAME password MYPASSWORD
# where 'MYUSERNAME' and 'MYPASSWORD' are your Earthdata credentials.
#
# Instead of a username/password, you may use an Earthdata bearer token.
# To construct a bearer token, log into Earthdata and choose "Generate Token".
# To use the token, when the script prompts for your username,
# just press Return (Enter). You will then be prompted for your token.
# You can store your bearer token in the .netrc file in the following format:
#    machine urs.earthdata.nasa.gov login token password MYBEARERTOKEN
# where 'MYBEARERTOKEN' is your Earthdata bearer token.
#
from __future__ import print_function

import base64
import getopt
import itertools
import json
import math
import netrc
import os.path
import ssl
import sys
import time
from getpass import getpass

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError, URLError, build_opener, HTTPCookieProcessor

short_name = ''
version = ''
time_start = ''
time_end = ''
bounding_box = ''
polygon = ''
filename_filter = ''
url_list = ['https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.10.20/ATL06_20201020111424_04010910_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.06.01/ATL06_20190601002242_09730312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.11.22/ATL06_20201122222952_09120912_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.09.28/ATL06_20190928183837_00280512_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.12.24/ATL06_20211224033709_00281412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.03.20/ATL06_20220320103739_13461410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.09.21/ATL06_20200921123821_13460810_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.10.24/ATL06_20201024235345_04700912_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.02.28/ATL06_20200228112206_09730612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.09.14/ATL06_20220914020547_12851610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.07.22/ATL06_20220722173304_04701612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.05.23/ATL06_20190523115139_08430310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.05.25/ATL06_20220525202047_09731512_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.09.18/ATL06_20220918015727_13461610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.03.16/ATL06_20220316104556_12851410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.09.15/ATL06_20210915192611_12851210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.12.17/ATL06_20201217082632_12850910_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.09.19/ATL06_20210919191751_13461210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.11.18/ATL06_20201118095027_08430910_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.03.02/ATL06_20190302044307_09730212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.05.23/ATL06_20210523134936_09121112_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.09.24/ATL06_20190924055913_13460410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.04.26/ATL06_20200426083413_04700712_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.02.26/ATL06_20190226045126_09120212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.11.21/ATL06_20211121050929_09121312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.02.17/ATL06_20210217053020_08431010_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.08.28/ATL06_20200828024142_09730812_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.11.27/ATL06_20181127091124_09120112_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.12.28/ATL06_20191228141823_00280612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.06.15/ATL06_20220615062550_12851510_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.09.17/ATL06_20200917124640_12850810_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.06.25/ATL06_20210625121718_00281212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.09.22/ATL06_20220922143651_00281712_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.01.22/ATL06_20200122001501_04010610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.02.24/ATL06_20220224004102_09731412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.10.23/ATL06_20191023043518_04010510_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.08.20/ATL06_20220820160908_09121612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.12.01/ATL06_20181201090306_09730112_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.05.19/ATL06_20210519011012_08431110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.01.23/ATL06_20210123193342_04701012_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.11.21/ATL06_20191121031119_08430510_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.10.23/ATL06_20211023063323_04701312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.02.25/ATL06_20210225180123_09731012_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.01.23/ATL06_20190123173555_04010210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.12.20/ATL06_20191220014720_12850510_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.05.29/ATL06_20200529070155_09730712_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.03.22/ATL06_20210322035803_13461010_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.01.19/ATL06_20210119065414_04011010_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.08.22/ATL06_20210822092930_09121212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.06.25/ATL06_20190625101919_13460310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.11.25/ATL06_20211125050110_09731312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.07.24/ATL06_20190724085517_04010410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.06.20/ATL06_20210620233755_13461110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.05.25/ATL06_20200525071015_09120712_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.07.26/ATL06_20200726041400_04700812_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.08.30/ATL06_20190830200231_09730412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.10.18/ATL06_20211018175357_04011310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.12.25/ATL06_20201225205736_00281012_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.02.24/ATL06_20200224113026_09120612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.04.21/ATL06_20200421195450_04010710_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.03.28/ATL06_20200328095806_00280712_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.08.24/ATL06_20200824025001_09120812_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.03.31/ATL06_20190331031906_00280312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.08.16/ATL06_20220816032944_08431610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.12.25/ATL06_20181225185948_13460110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.08.26/ATL06_20190826201051_09120412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.06.27/ATL06_20200627053756_00280812_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.09.24/ATL06_20210924075716_00281312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.07.18/ATL06_20220718045340_04011610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.07.24/ATL06_20210724105323_04701212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.02.21/ATL06_20190221161201_08430210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.11.22/ATL06_20181122203207_08430110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.12.15/ATL06_20211215150608_12851310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.04.24/ATL06_20190424131542_04010310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.02.19/ATL06_20200219225103_08430610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.03.19/ATL06_20200319212707_12850610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.06.18/ATL06_20200618170652_12850710_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.01.17/ATL06_20220117133351_04011410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.09.20/ATL06_20190920060731_12850410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.03.26/ATL06_20190326143942_13460210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.12.21/ATL06_20201221081813_13460910_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.03.24/ATL06_20220324231701_00281512_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.05.17/ATL06_20220517074946_08431510_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.07.21/ATL06_20200721153436_04010810_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.06.23/ATL06_20220623185704_00281612_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.12.21/ATL06_20181221190808_12850110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.08.19/ATL06_20200819141038_08430810_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.05.21/ATL06_20220521202906_09121512_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.06.16/ATL06_20210616234614_12851110_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.07.28/ATL06_20190728213443_04700412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.06.19/ATL06_20220619061733_13461510_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.02.20/ATL06_20220220004917_09121412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.04.24/ATL06_20210424151331_04701112_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.04.22/ATL06_20220422215308_04701512_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.12.24/ATL06_20191224013900_13460510_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.02.21/ATL06_20210221180945_09121012_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.01.28/ATL06_20190128061524_04700212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.05.28/ATL06_20190528003102_09120312_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.06.22/ATL06_20200622165833_13460710_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.12.19/ATL06_20211219145748_13461310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.10.27/ATL06_20191027171440_04700512_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2018.12.30/ATL06_20181230073914_00280212_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.08.17/ATL06_20210817205006_08431210_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.03.23/ATL06_20200323211844_13460610_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.03.18/ATL06_20210318040622_12851010_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.01.22/ATL06_20220122021314_04701412_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2019.08.22/ATL06_20190822073127_08430410_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.04.18/ATL06_20220418091333_04011510_005_02.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.01.26/ATL06_20200126125426_04700612_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.05.27/ATL06_20210527134115_09731112_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2020.09.26/ATL06_20200926011745_00280912_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2021.11.16/ATL06_20211116163005_08431310_005_01.h5',
            'https://n5eil01u.ecs.nsidc.org/DP7/ATLAS/ATL06.005/2022.08.24/ATL06_20220824160051_09731612_005_01.h5']

CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                '&scroll=true&page_size={1}'.format(CMR_URL, CMR_PAGE_SIZE))


def get_username():
    username = ''

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    username = do_input('Earthdata username (or press Return to use a bearer token): ')
    return username


def get_password():
    password = ''
    while not password:
        password = getpass('password: ')
    return password


def get_token():
    token = ''
    while not token:
        token = getpass('bearer token: ')
    return token


def get_login_credentials():
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    token = None

    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        if username == 'token':
            token = password
        else:
            credentials = '{0}:{1}'.format(username, password)
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
    except Exception:
        username = None
        password = None

    if not username:
        username = get_username()
        if len(username):
            password = get_password()
            credentials = '{0}:{1}'.format(username, password)
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
        else:
            token = get_token()

    return credentials, token


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ''

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += '&version={0}'.format(padded_version)
        desired_pad_length -= 1
    return query_params


def filter_add_wildcards(filter):
    if not filter.startswith('*'):
        filter = '*' + filter
    if not filter.endswith('*'):
        filter = filter + '*'
    return filter


def build_filename_filter(filename_filter):
    filters = filename_filter.split(',')
    result = '&options[producer_granule_id][pattern]=true'
    for filter in filters:
        result += '&producer_granule_id[]=' + filter_add_wildcards(filter)
    return result


def build_cmr_query_url(short_name, version, time_start, time_end,
                        bounding_box=None, polygon=None,
                        filename_filter=None):
    params = '&short_name={0}'.format(short_name)
    params += build_version_query_params(version)
    params += '&temporal[]={0},{1}'.format(time_start, time_end)
    if polygon:
        params += '&polygon={0}'.format(polygon)
    elif bounding_box:
        params += '&bounding_box={0}'.format(bounding_box)
    if filename_filter:
        params += build_filename_filter(filename_filter)
    return CMR_FILE_URL + params


def get_speed(time_elapsed, chunk_size):
    if time_elapsed <= 0:
        return ''
    speed = chunk_size / time_elapsed
    if speed <= 0:
        speed = 1
    size_name = ('', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    i = int(math.floor(math.log(speed, 1000)))
    p = math.pow(1000, i)
    return '{0:.1f}{1}B/s'.format(speed / p, size_name[i])


def output_progress(count, total, status='', bar_len=60):
    if total <= 0:
        return
    fraction = min(max(count / float(total), 0), 1)
    filled_len = int(round(bar_len * fraction))
    percents = int(round(100.0 * fraction))
    bar = '=' * filled_len + ' ' * (bar_len - filled_len)
    fmt = '  [{0}] {1:3d}%  {2}   '.format(bar, percents, status)
    print('\b' * (len(fmt) + 4), end='')  # clears the line
    sys.stdout.write(fmt)
    sys.stdout.flush()


def cmr_read_in_chunks(file_object, chunk_size=1024 * 1024):
    """Read a file in chunks using a generator. Default chunk size: 1Mb."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_login_response(url, credentials, token):
    opener = build_opener(HTTPCookieProcessor())

    req = Request(url)
    if token:
        req.add_header('Authorization', 'Bearer {0}'.format(token))
    elif credentials:
        try:
            response = opener.open(req)
            # We have a redirect URL - try again with authorization.
            url = response.url
        except HTTPError:
            # No redirect - just try again with authorization.
            pass
        except Exception as e:
            print('Error{0}: {1}'.format(type(e), str(e)))
            sys.exit(1)

        req = Request(url)
        req.add_header('Authorization', 'Basic {0}'.format(credentials))

    try:
        response = opener.open(req)
    except HTTPError as e:
        err = 'HTTP error {0}, {1}'.format(e.code, e.reason)
        if 'Unauthorized' in e.reason:
            if token:
                err += ': Check your bearer token'
            else:
                err += ': Check your username and password'
        print(err)
        sys.exit(1)
    except Exception as e:
        print('Error{0}: {1}'.format(type(e), str(e)))
        sys.exit(1)

    return response


def cmr_download(urls, force=False, quiet=False):
    """Download files from list of urls."""
    if not urls:
        return

    url_count = len(urls)
    if not quiet:
        print('Downloading {0} files...'.format(url_count))
    credentials = None
    token = None

    for index, url in enumerate(urls, start=1):
        if not credentials and not token:
            p = urlparse(url)
            if p.scheme == 'https':
                credentials, token = get_login_credentials()

        filename = url.split('/')[-1]
        if not quiet:
            print('{0}/{1}: {2}'.format(str(index).zfill(len(str(url_count))),
                                        url_count, filename))

        try:
            response = get_login_response(url, credentials, token)
            length = int(response.headers['content-length'])
            try:
                if not force and length == os.path.getsize(filename):
                    if not quiet:
                        print('  File exists, skipping')
                    continue
            except OSError:
                pass
            count = 0
            chunk_size = min(max(length, 1), 1024 * 1024)
            max_chunks = int(math.ceil(length / chunk_size))
            time_initial = time.time()
            with open(filename, 'wb') as out_file:
                for data in cmr_read_in_chunks(response, chunk_size=chunk_size):
                    out_file.write(data)
                    if not quiet:
                        count = count + 1
                        time_elapsed = time.time() - time_initial
                        download_speed = get_speed(time_elapsed, count * chunk_size)
                        output_progress(count, max_chunks, status=download_speed)
            if not quiet:
                print()
        except HTTPError as e:
            print('HTTP error {0}, {1}'.format(e.code, e.reason))
        except URLError as e:
            print('URL error: {0}'.format(e.reason))
        except IOError:
            raise


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if 'feed' not in search_results or 'entry' not in search_results['feed']:
        return []

    entries = [e['links']
               for e in search_results['feed']['entry']
               if 'links' in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if 'href' not in link:
            # Exclude links with nothing to download
            continue
        if 'inherited' in link and link['inherited'] is True:
            # Why are we excluding these links?
            continue
        if 'rel' in link and 'data#' not in link['rel']:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if 'title' in link and 'opendap' in link['title'].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link['href'].split('/')[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link['href'])

    return urls


def cmr_search(short_name, version, time_start, time_end,
               bounding_box='', polygon='', filename_filter='', quiet=False):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(short_name=short_name, version=version,
                                        time_start=time_start, time_end=time_end,
                                        bounding_box=bounding_box,
                                        polygon=polygon, filename_filter=filename_filter)
    if not quiet:
        print('Querying for data:\n\t{0}\n'.format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = []
    hits = 0
    while True:
        req = Request(cmr_query_url)
        if cmr_scroll_id:
            req.add_header('cmr-scroll-id', cmr_scroll_id)
        try:
            response = urlopen(req, context=ctx)
        except Exception as e:
            print('Error: ' + str(e))
            sys.exit(1)
        if not cmr_scroll_id:
            # Python 2 and 3 have different case for the http headers
            headers = {k.lower(): v for k, v in dict(response.info()).items()}
            cmr_scroll_id = headers['cmr-scroll-id']
            hits = int(headers['cmr-hits'])
            if not quiet:
                if hits > 0:
                    print('Found {0} matches.'.format(hits))
                else:
                    print('Found no matches.')
        search_page = response.read()
        search_page = json.loads(search_page.decode('utf-8'))
        url_scroll_results = cmr_filter_urls(search_page)
        if not url_scroll_results:
            break
        if not quiet and hits > CMR_PAGE_SIZE:
            print('.', end='')
            sys.stdout.flush()
        urls += url_scroll_results

    if not quiet and hits > CMR_PAGE_SIZE:
        print()
    return urls


def main(argv=None):
    global short_name, version, time_start, time_end, bounding_box, \
        polygon, filename_filter, url_list

    if argv is None:
        argv = sys.argv[1:]

    force = False
    quiet = False
    usage = 'usage: nsidc-download_***.py [--help, -h] [--force, -f] [--quiet, -q]'

    try:
        opts, args = getopt.getopt(argv, 'hfq', ['help', 'force', 'quiet'])
        for opt, _arg in opts:
            if opt in ('-f', '--force'):
                force = True
            elif opt in ('-q', '--quiet'):
                quiet = True
            elif opt in ('-h', '--help'):
                print(usage)
                sys.exit(0)
    except getopt.GetoptError as e:
        print(e.args[0])
        print(usage)
        sys.exit(1)

    # Supply some default search parameters, just for testing purposes.
    # These are only used if the parameters aren't filled in up above.
    if 'short_name' in short_name:
        short_name = 'ATL06'
        version = '003'
        time_start = '2018-10-14T00:00:00Z'
        time_end = '2021-01-08T21:48:13Z'
        bounding_box = ''
        polygon = ''
        filename_filter = '*ATL06_2020111121*'
        url_list = []

    try:
        if not url_list:
            url_list = cmr_search(short_name, version, time_start, time_end,
                                  bounding_box=bounding_box, polygon=polygon,
                                  filename_filter=filename_filter, quiet=quiet)

        cmr_download(url_list, force=force, quiet=quiet)
    except KeyboardInterrupt:
        quit()


if __name__ == '__main__':
    main()
