import os

import pytest
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue

from clisops.ops.average import average_over_dims, average_time

from .._common import C3S_CORDEX_EUR_ZG500, CMIP5_TAS, CMIP6_SICONC_DAY


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath, use_cftime=True)


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


def test_average_no_dims():
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
        == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude', 'realization']"
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


def test_average_over_years():
    ds = _load_ds(CMIP5_TAS)  # monthly dataset

    # check initial dataset
    assert ds.time.shape == (3530,)
    assert ds.time.values[0].isoformat() == "2005-12-16T00:00:00"
    assert ds.time.values[-1].isoformat() == "2299-12-16T00:00:00"

    result = average_time(
        CMIP5_TAS,
        freq="year",
        output_type="xarray",
    )

    time_length = ds.time.values[-1].year - ds.time.values[0].year + 1
    assert result[0].time.shape == (time_length,)  # get number of years
    assert result[0].time.values[0].isoformat() == "2005-01-01T00:00:00"
    assert result[0].time.values[-1].isoformat() == "2299-01-01T00:00:00"

    # test time bounds
    assert [t.isoformat() for t in result[0].time_bnds.values[0]] == [
        "2005-01-01T00:00:00",
        "2005-12-30T00:00:00",
    ]
    assert [t.isoformat() for t in result[0].time_bnds.values[-1]] == [
        "2299-01-01T00:00:00",
        "2299-12-30T00:00:00",
    ]


def test_average_over_months():
    ds = _load_ds(CMIP6_SICONC_DAY)  # monthly dataset

    # check initial dataset
    assert ds.time.shape == (60225,)
    assert ds.time.values[0].isoformat() == "1850-01-01T12:00:00"
    assert ds.time.values[-1].isoformat() == "2014-12-31T12:00:00"

    # average over time
    result = average_time(
        CMIP6_SICONC_DAY,
        freq="month",
        output_type="xarray",
    )

    time_length = (
        ds.time.values[-1].year - ds.time.values[0].year + 1
    ) * 12  # get number of months

    assert result[0].time.shape == (time_length,)
    assert result[0].time.values[0].isoformat() == "1850-01-01T00:00:00"
    assert result[0].time.values[-1].isoformat() == "2014-12-01T00:00:00"

    # test time bounds
    assert [t.isoformat() for t in result[0].time_bnds.values[0]] == [
        "1850-01-01T00:00:00",
        "1850-01-31T00:00:00",
    ]
    assert [t.isoformat() for t in result[0].time_bnds.values[-1]] == [
        "2014-12-01T00:00:00",
        "2014-12-31T00:00:00",
    ]


def test_average_time_no_freq():
    with pytest.raises(InvalidParameterValue) as exc:
        # average over time
        average_time(
            CMIP6_SICONC_DAY,
            freq=None,  # noqa
            output_type="xarray",
        )
    assert str(exc.value) == "At least one frequency for averaging must be provided"


def test_average_time_incorrect_freq():
    with pytest.raises(InvalidParameterValue) as exc:
        # average over time
        average_time(
            CMIP6_SICONC_DAY,
            freq="week",
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "Time frequency for averaging must be one of ['day', 'month', 'year']."
    )


def test_average_time_file_name(tmpdir):
    result = average_time(
        CMIP5_TAS,
        freq="year",
        output_type="nc",
        output_dir=tmpdir,
    )

    _check_output_nc(
        result, fname="tas_mon_HadGEM2-ES_rcp85_r1i1p1_20050101-22990101_avg-year.nc"
    )


def test_average_time_cordex():
    ds = _load_ds(C3S_CORDEX_EUR_ZG500)

    # check initial dataset
    assert ds.time.shape == (3653,)
    assert ds.time.values[0].isoformat() == "2071-01-01T12:00:00"
    assert ds.time.values[-1].isoformat() == "2080-12-31T12:00:00"

    # average over time
    result = average_time(
        C3S_CORDEX_EUR_ZG500,
        freq="month",
        output_type="xarray",
    )

    time_length = (
        ds.time.values[-1].year - ds.time.values[0].year + 1
    ) * 12  # get number of months

    assert result[0].time.shape == (time_length,)
    assert result[0].time.values[0].isoformat() == "2071-01-01T00:00:00"
    assert result[0].time.values[-1].isoformat() == "2080-12-01T00:00:00"

    # test time bounds
    assert [t.isoformat() for t in result[0].time_bnds.values[0]] == [
        "2071-01-01T00:00:00",
        "2071-01-31T00:00:00",
    ]
    assert [t.isoformat() for t in result[0].time_bnds.values[-1]] == [
        "2080-12-01T00:00:00",
        "2080-12-31T00:00:00",
    ]
