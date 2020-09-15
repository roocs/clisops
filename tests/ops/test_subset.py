import os
import sys
from unittest.mock import Mock

import pytest
from memory_profiler import memory_usage
from roocs_utils.exceptions import InvalidParameterValue, MissingParameterValue
from roocs_utils.parameter import area_parameter, time_parameter
from roocs_utils.utils.common import parse_size

import clisops
from clisops import CONFIG
from clisops.ops.subset import subset
from clisops.utils import output_utils
from clisops.utils.output_utils import _format_time, get_time_slices

from .._common import CMIP5_RH, CMIP5_TAS, CMIP5_TAS_FILE, CMIP5_ZOSTOGA


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


@pytest.mark.xfail(
    reason="Time, Level and area can all be none as they default to max/min values"
    "in core.subset"
)
def test_subset_missing_param(tmpdir):
    """ Test subset without area param."""
    with pytest.raises(MissingParameterValue):
        subset(ds=CMIP5_TAS_FILE, output_dir=tmpdir)


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
        area=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_area_project_file_name(tmpdir):
    """ Tests clisops subset function with a area subset (derived file name)."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        area=(0.0, 49.0, 10.0, 65.0),
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
        area=(0.0, 49.0, 10.0, 65.0),
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
        area=(0.0, 49.0, 10.0, 65.0),
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
        area=(0.0, 49.0, 10.0, 65.0),
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
        area=(0.0, 49.0, 10.0, 65.0),
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


def subset_for_test(real_data, start_time, end_time):
    subset(
        ds=real_data,
        time=(start_time, end_time),
        area=(0.0, 49.0, 10.0, 65.0),
        output_type="xarray",
        file_namer="simple",
    )


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_memory_limit():
    """ check memory does not greatly exceed dask chunk limit """

    real_data = (
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES"
        "/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"
    )

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    config_mem_limit = CONFIG["clisops:read"]["chunk_memory_limit"]

    CONFIG["clisops:write"]["file_size_limit"] = "95MiB"
    CONFIG["clisops:read"]["chunk_memory_limit"] = "80MiB"

    memory = memory_usage((subset_for_test, (real_data, start_time, end_time)))

    upper_limit = 80 * 1.1

    assert max(memory) <= upper_limit

    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size
    CONFIG["clisops:read"]["chunk_memory_limit"] = config_mem_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_bigger_file():
    real_data = (
        "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11/IPSL/MOHC-HadGEM2-ES/rcp85"
        "/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc"
    )

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    config_mem_limit = CONFIG["clisops:read"]["chunk_memory_limit"]
    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]

    CONFIG["clisops:write"]["file_size_limit"] = "750MiB"
    CONFIG["clisops:read"]["chunk_memory_limit"] = "100MiB"

    memory = memory_usage((subset_for_test, (real_data, start_time, end_time)))

    upper_limit = 100 * 1.1

    assert max(memory) <= upper_limit

    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size
    CONFIG["clisops:read"]["chunk_memory_limit"] = config_mem_limit


@pytest.fixture(params=["0.4MiB", "45MiB", "570MiB", "45MiB"])
def mem_limit(request):
    id = request.param
    return id


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_even_bigger_file(mem_limit):
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    real_data = (
        "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES/rcp85/3hr"
        "/atmos/3hr/r1i1p1/vas/v20111002/*.nc"
    )

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (real_data, start_time, end_time)))
    upper_limit = 500

    assert max(memory) <= upper_limit
