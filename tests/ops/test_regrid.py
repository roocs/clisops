import os
import sys
from unittest.mock import Mock

import pytest
import xarray as xr

import clisops
from clisops import CONFIG
from clisops.ops.regrid import regrid

from .._common import CMIP5_TAS


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_regrid_none(tmpdir):
    """ test behaviour when none passed as method and grid - should the default regrdding take place?"""
    result = regrid(
        CMIP5_TAS,
        method=None,
        grid=None,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_regrid-nn-1deg.nc")


def test_regrid_basic(tmpdir):
    """ test a basic regridding oepration"""
    result = regrid(
        CMIP5_TAS,
        method="nn",
        adaptive_masking_threshold=0.5,
        grid="1deg",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_regrid-nn-1deg.nc")
