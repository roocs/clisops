import os

import pytest
from roocs_utils.exceptions import InvalidParameterValue, MissingParameterValue
from roocs_utils.parameter import area_parameter, time_parameter

from clisops.ops.subset import subset

from .._common import CMIP5_RH, CMIP5_TAS, CMIP5_TAS_FILE, CMIP5_ZOSTOGA


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
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_args_as_parameter_classes(tmpdir):
    """Tests clisops subset function with a time subset
    with the arguments as parameter classes from roocs-utils."""

    time = time_parameter.TimeParameter(("2020-01-01T00:00:00", "2020-12-30T00:00:00"))
    area = area_parameter.AreaParameter((0, -90.0, 360.0, 90.0))

    result = subset(
        ds=CMIP5_TAS_FILE,
        time=time,
        area=area,
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_invalid_time(tmpdir):
    """ Tests subset with invalid time param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=CMIP5_TAS_FILE,
            time=("yesterday", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
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


def test_subset_area(tmpdir):
    """ Tests clisops subset function with a area subset."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        area=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


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
    )
    assert "output.nc" in result


def test_subset_with_time_and_area(tmpdir):
    """ Tests clisops subset function with time, area, level subsets."""
    result = subset(
        ds=CMIP5_TAS_FILE,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_tas(tmpdir):
    """ Tests with multiple tas files"""
    result = subset(
        ds=CMIP5_TAS,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_zostoga(tmpdir):
    """ Tests with multiple tas files"""
    result = subset(
        ds=CMIP5_ZOSTOGA,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_rh(tmpdir):
    """ Tests with multiple rh files"""
    result = subset(
        ds=CMIP5_RH,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_tas_series(tmpdir, tas_series):
    """ Test with tas_series fixture"""
    result = subset(
        ds=tas_series(["20", "22", "25"]),
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result
