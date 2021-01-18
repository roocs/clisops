import os
import sys
from unittest.mock import Mock

import xarray as xr

import clisops
from clisops import CONFIG
from clisops.ops.average import average_over_dims


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_average_basic(cmip5_tas, tmpdir):
    result = average_over_dims(
        cmip5_tas,
        dims=None,
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216.nc"
    )


def test_average_time(cmip5_tas):
    result = average_over_dims(
        cmip5_tas, dims=["time"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "time" not in result[0]


def test_average_lat(cmip5_tas):
    result = average_over_dims(
        cmip5_tas, dims=["latitude"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "lat" not in result[0]


def test_average_lon(cmip5_tas):
    result = average_over_dims(
        cmip5_tas, dims=["longitude"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "lon" not in result[0]


def test_average_level(cmip5_tas):
    result = average_over_dims(
        cmip5_tas, dims=["level"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "level" not in result[0]
