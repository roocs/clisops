import os
import sys
from unittest.mock import Mock

import xarray as xr

import clisops
from clisops import CONFIG
from clisops.ops.average import average_over_dims

from .._common import CMIP5_TAS


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_average_basic(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=None,
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216.nc"
    )


def test_average_basic_data_array(cmip5_tas_file):
    ds = xr.open_dataset(cmip5_tas_file)
    result = average_over_dims(
        ds["tas"], dims=["time"], ignore_unfound_dims=False, output_type="xarray"
    )
    assert "time" not in result[0]


def test_average_time_xarray():
    result = average_over_dims(
        CMIP5_TAS, dims=["time"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "time" not in result[0]


def test_average_lat_xarray():
    result = average_over_dims(
        CMIP5_TAS, dims=["latitude"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "lat" not in result[0]


def test_average_lon_xarray():
    result = average_over_dims(
        CMIP5_TAS, dims=["longitude"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "lon" not in result[0]


def test_average_level_xarray(cmip6_o3):
    result = average_over_dims(
        cmip6_o3, dims=["level"], ignore_unfound_dims=False, output_type="xarray"
    )

    assert "plev" not in result[0]


def test_average_time_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time"],
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg_time.nc")


def test_average_lat_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["latitude"],
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg_latitude.nc")


def test_average_lon_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["longitude"],
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg_longitude.nc")


def test_average_level_nc(cmip6_o3, tmpdir):
    result = average_over_dims(
        cmip6_o3,
        dims=["level"],
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_avg_level.nc"
    )


def test_average_multiple_dims_filename(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time", "longitude"],
        ignore_unfound_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg_longitude-time.nc"
    )


def test_average_multiple_dims_xarray():
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time", "longitude"],
        ignore_unfound_dims=False,
        output_type="xarray",
    )

    assert "time" not in result[0]
    assert "lon" not in result[0]
