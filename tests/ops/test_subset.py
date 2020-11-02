import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue, MissingParameterValue
from roocs_utils.parameter import area_parameter, time_parameter
from roocs_utils.utils.common import parse_size

import clisops
from clisops import CONFIG
from clisops.ops.subset import _subset, subset
from clisops.utils import map_params, output_utils
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import _format_time, get_output, get_time_slices

from .._common import (
    C3S_CMIP5_TOS,
    C3S_CMIP5_TSICE,
    CMIP5_RH,
    CMIP5_TAS,
    CMIP5_TAS_FILE,
    CMIP5_ZOSTOGA,
    CMIP6_O3,
)


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_subset_no_params(tmpdir):
    """ Test subset without area param."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_time(tmpdir):
    """ Tests clisops subset function with a time subset."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_args_as_parameter_classes(tmpdir):
    """Tests clisops subset function with a time subset
    with the arguments as parameter classes from roocs-utils."""

    time = time_parameter.TimeParameter(("2000-01-01T00:00:00", "2020-12-30T00:00:00"))
    area = area_parameter.AreaParameter((0, -90.0, 360.0, 90.0))

    result = subset(
        ds=CMIP5_TAS_FILE,
        time=time,
        area=area,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_invalid_time(tmpdir):
    """ Tests subset with invalid time param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=CMIP5_TAS_FILE,
            time=("yesterday", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )


def test_subset_ds_is_none(tmpdir):
    """ Tests subset with ds=None."""
    with pytest.raises(MissingParameterValue):
        subset(
            ds=None,
            time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_no_ds(tmpdir):
    """ Tests subset with no dataset provided."""
    with pytest.raises(TypeError):
        subset(
            time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_area_simple_file_name(tmpdir):
    """ Tests clisops subset function with a area subset (simple file name)."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_area_project_file_name(tmpdir):
    """ Tests clisops subset function with a area subset (derived file name)."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )
    _check_output_nc(result, "tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-20301116.nc")


def test_subset_invalid_area(tmpdir):
    """ Tests subset with invalid area param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=CMIP5_TAS_FILE,
            area=("zero", 49.0, 10.0, 65.0),
            output_dir=tmpdir,
        )


@pytest.mark.xfail(reason="cross the 0 degree meridian not implemented.")
def test_subset_area_with_meridian(tmpdir):
    """ Tests clisops subset function with a area subset."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        area=(-10.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_time_and_area(tmpdir):
    """ Tests clisops subset function with time, area, level subsets."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        time=("2019-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 0.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_multiple_files_tas(tmpdir):
    """ Tests with multiple tas files"""
    result = subset(
        ds=CMIP5_TAS,
        time=("2001-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 0.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_multiple_files_zostoga(tmpdir):
    """ Tests with multiple zostoga files"""
    result = subset(
        ds=CMIP5_ZOSTOGA,
        time=("2000-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_multiple_files_rh(tmpdir):
    """ Tests with multiple rh files"""
    result = subset(
        ds=CMIP5_RH,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_tas_series(tmpdir, tas_series):
    """ Test with tas_series fixture"""
    result = subset(
        ds=tas_series(["20", "22", "25"]),
        time=("2000-07-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_time_slices_in_subset_tas():
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    time_slices = [
        ("2005-12-16", "2040-03-16"),
        ("2040-04-16", "2074-07-16"),
        ("2074-08-16", "2108-10-16"),
        ("2108-11-16", "2143-02-16"),
        ("2143-03-16", "2177-06-16"),
        ("2177-07-16", "2199-12-16"),
    ]

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size

    outputs = subset(
        ds=CMIP5_TAS,
        time=(start_time, end_time),
        area=(0.0, 5.0, 50.0, 90.0),
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    assert _format_time(outputs[0].time.values.min()) >= start_time
    assert _format_time(outputs[-1].time.values.max()) <= end_time

    count = 0
    for _ in outputs:
        assert _format_time(outputs[count].time.values.min()) >= time_slices[count][0]
        assert _format_time(outputs[count].time.values.max()) >= time_slices[count][1]
        count += 1


def test_time_slices_in_subset_rh():
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    time_slices = [
        ("2001-01-16", "2002-09-16"),
        ("2002-10-16", "2004-06-16"),
        ("2004-07-16", "2005-11-16"),
    ]

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=CMIP5_RH,
        time=(start_time, end_time),
        area=(0.0, 5.0, 50.0, 90.0),
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    assert _format_time(outputs[0].time.values.min()) >= start_time
    assert _format_time(outputs[-1].time.values.max()) <= end_time

    count = 0
    for _ in outputs:
        assert _format_time(outputs[count].time.values.min()) >= time_slices[count][0]
        assert _format_time(outputs[count].time.values.max()) >= time_slices[count][1]
        count += 1


# area can be a few degrees out
def test_area_within_area_subset():
    area = (0.0, 10.0, 175.0, 90.0)

    outputs = subset(
        ds=CMIP5_TAS,
        time=("2001-01-01T00:00:00", "2200-12-30T00:00:00"),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )

    ds = outputs[0]
    assert area[0] <= ds.lon.data <= area[2]
    assert area[1] <= ds.lat.data <= area[3]


def test_area_within_area_subset_chunked():

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
    area = (0.0, 10.0, 175.0, 90.0)

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=CMIP5_TAS,
        time=(start_time, end_time),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    for ds in outputs:
        assert area[0] <= ds.lon.data <= area[2]
        assert area[1] <= ds.lat.data <= area[3]


def test_subset_level(tmpdir):
    """ Tests clisops subset function with a level subset."""
    # Levels are: 100000, ..., 100
    ds = _load_ds(CMIP6_O3)

    result1 = subset(ds=CMIP6_O3, level="100000/100", output_type="xarray")

    np.testing.assert_array_equal(result1[0].o3.values, ds.o3.values)

    result2 = subset(ds=CMIP6_O3, level="100/100", output_type="xarray")

    np.testing.assert_array_equal(result2[0].o3.shape, (1200, 1, 2, 3))

    result3 = subset(ds=CMIP6_O3, level="101/-23.234", output_type="xarray")

    np.testing.assert_array_equal(result3[0].o3.values, result2[0].o3.values)


def test_aux_variables():
    """
    test auxiliary variables are remembered in output dataset
    Have to create a netcdf file with auxiliary variable
    """

    ds = _load_ds("tests/ops/file.nc")

    assert "do_i_get_written" in ds.variables

    result = subset(
        ds=ds,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "do_i_get_written" in result[0].variables


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_coord_variables_exist():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid exist in output dataset
    """
    ds = _load_ds(C3S_CMIP5_TSICE)

    assert "lat" in ds.coords
    assert "lon" in ds.coords

    result = subset(
        ds=C3S_CMIP5_TSICE,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "lat" in result[0].coords
    assert "lon" in result[0].coords


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_coord_variables_subsetted_i_j():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """

    ds = _load_ds(C3S_CMIP5_TSICE)

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "i" in ds.dims
    assert "j" in ds.dims

    area = (5.0, 10.0, 20.0, 65.0)

    result = subset(
        ds=C3S_CMIP5_TSICE,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    # check within 10% of expected subset value
    assert abs(area[1] - float(result[0].lat.min())) / area[1] <= 0.1
    assert abs(float(result[0].lat.max()) - area[3]) / area[3] <= 0.1

    with pytest.raises(AssertionError):
        assert abs(area[0] - float(result[0].lon.min())) / area[0] <= 0.1
        assert abs(float(result[0].lon.max()) - area[2]) / area[2] <= 0.1
        # working for lat but not lon in this example


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_coord_variables_subsetted_rlat_rlon():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """

    ds = _load_ds(C3S_CMIP5_TOS)

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "rlat" in ds.dims
    assert "rlon" in ds.dims

    area = (5.0, 10.0, 20.0, 65.0)

    result = subset(
        ds=C3S_CMIP5_TOS,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    # check within 10% of expected subset value
    assert abs(area[1] - float(result[0].lat.min())) / area[1] <= 0.1
    assert abs(float(result[0].lat.max()) - area[3]) / area[3] <= 0.1
    assert abs(area[0] - float(result[0].lon.min())) / area[0] <= 0.1
    assert abs(float(result[0].lon.max()) - area[2]) / area[2] <= 0.1
