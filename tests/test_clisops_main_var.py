import os

from clisops.utils import get_coords
from ._common import CMIP5_ARCHIVE_BASE

import xarray as xr


CMIP5_FPATHS = [
    os.path.join(CMIP5_ARCHIVE_BASE, 'cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/*.nc'),
    os.path.join(CMIP5_ARCHIVE_BASE, 'cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc'),
    os.path.join(CMIP5_ARCHIVE_BASE, 'cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc')
]


def test_get_main_var_1():
    ds = xr.open_mfdataset(CMIP5_FPATHS[0])
    var_id = get_coords.get_main_variable(ds)
    assert var_id == 'zostoga'


def test_get_main_var_2():
    ds = xr.open_mfdataset(CMIP5_FPATHS[1])
    var_id = get_coords.get_main_variable(ds)
    assert var_id == 'tas'


def test_get_main_var_3():
    ds = xr.open_mfdataset(CMIP5_FPATHS[2])
    var_id = get_coords.get_main_variable(ds)
    assert var_id == 'rh'
