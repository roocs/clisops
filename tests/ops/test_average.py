import os
import sys
from unittest.mock import Mock

import pytest
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue

import clisops
from clisops import CONFIG
from clisops.ops.average import average_over_dims

from .._common import CMIP5_TAS


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_average_basic_data_array(cmip5_tas_file):
    ds = xr.open_dataset(cmip5_tas_file)
    result = average_over_dims(
        ds["tas"], dims=["time"], ignore_undetected_dims=False, output_type="xarray"
    )
    assert "time" not in result[0]


def test_average_time_xarray():
    result = average_over_dims(
        CMIP5_TAS, dims=["time"], ignore_undetected_dims=False, output_type="xarray"
    )

    assert "time" not in result[0]


def test_average_lat_xarray():
    result = average_over_dims(
        CMIP5_TAS, dims=["latitude"], ignore_undetected_dims=False, output_type="xarray"
    )

    assert "lat" not in result[0]


def test_average_lon_xarray():
    result = average_over_dims(
        CMIP5_TAS,
        dims=["longitude"],
        ignore_undetected_dims=False,
        output_type="xarray",
    )

    assert "lon" not in result[0]


def test_average_level_xarray(cmip6_o3):
    result = average_over_dims(
        cmip6_o3, dims=["level"], ignore_undetected_dims=False, output_type="xarray"
    )

    assert "plev" not in result[0]


def test_average_time_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time"],
        ignore_undetected_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )
    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg-t.nc")


def test_average_lat_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["latitude"],
        ignore_undetected_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216_avg-y.nc"
    )


def test_average_lon_nc(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["longitude"],
        ignore_undetected_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216_avg-x.nc"
    )


def test_average_level_nc(cmip6_o3, tmpdir):
    result = average_over_dims(
        cmip6_o3,
        dims=["level"],
        ignore_undetected_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result,
        fname="o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_18500116-19491216_avg-z.nc",
    )


def test_average_multiple_dims_filename(tmpdir):
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time", "longitude"],
        ignore_undetected_dims=False,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_avg-tx.nc")


def test_average_multiple_dims_xarray():
    result = average_over_dims(
        CMIP5_TAS,
        dims=["time", "longitude"],
        ignore_undetected_dims=False,
        output_type="xarray",
    )

    assert "time" not in result[0]
    assert "lon" not in result[0]


def test_average_no_dims(tmpdir):
    with pytest.raises(InvalidParameterValue) as exc:
        average_over_dims(
            CMIP5_TAS,
            dims=None,
            ignore_undetected_dims=False,
            output_type="xarray",
        )
    assert str(exc.value) == "At least one dimension for averaging must be provided"


def test_unknown_dim():
    with pytest.raises(InvalidParameterValue) as exc:
        average_over_dims(
            CMIP5_TAS,
            dims=["wrong"],
            ignore_undetected_dims=False,
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude']"
    )


def test_dim_not_found():
    with pytest.raises(InvalidParameterValue) as exc:
        average_over_dims(
            CMIP5_TAS,
            dims=["level", "time"],
            ignore_undetected_dims=False,
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "Requested dimensions were not found in input dataset: {'level'}."
    )


def test_dim_not_found_ignore():
    # cannot average over level, but have ignored it and done average over time anyway
    result = average_over_dims(
        CMIP5_TAS,
        dims=["level", "time"],
        ignore_undetected_dims=True,
        output_type="xarray",
    )

    assert "time" not in result[0]
    assert "height" in result[0]


def test_aux_variables():
    """
    test auxiliary variables are remembered in output dataset
    Have to create a netcdf file with auxiliary variable
    """

    ds = _load_ds("tests/ops/file.nc")

    assert "do_i_get_written" in ds.variables

    result = average_over_dims(
        ds=ds,
        dims=["level", "time"],
        ignore_undetected_dims=True,
        output_type="xarray",
    )

    assert "do_i_get_written" in result[0].variables
