import os

import pytest

from clisops.exceptions import InvalidParameterValue, MissingParameterValue
from clisops.ops.subset import subset

from .._common import CMIP5_RH, CMIP5_TAS, CMIP5_TAS_FILE, CMIP5_ZOSTOGA


def test_subset_missing_param(tmpdir):
    """ Test subset without time or space param."""
    with pytest.raises(MissingParameterValue):
        subset(
            dset=CMIP5_TAS_FILE, output_dir=tmpdir,
        )


def test_subset_time(tmpdir):
    """ Tests clisops subset function with a time subset."""
    result = subset(
        dset=CMIP5_TAS_FILE,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_invalid_time(tmpdir):
    """ Tests subset with invalid time param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            dset=CMIP5_TAS_FILE,
            time=("yesterday", "2020-12-30T00:00:00"),
            output_dir=tmpdir,
        )


def test_subset_space(tmpdir):
    """ Tests clisops subset function with a space subset."""
    result = subset(
        dset=CMIP5_TAS_FILE, space=(0.0, 49.0, 10.0, 65.0), output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_invalid_space(tmpdir):
    """ Tests subset with invalid space param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            dset=CMIP5_TAS_FILE, space=("zero", 49.0, 10.0, 65.0), output_dir=tmpdir,
        )


@pytest.mark.xfail(reason="cross the 0 degree meridian not implemented.")
def test_subset_space_with_meridian(tmpdir):
    """ Tests clisops subset function with a space subset."""
    result = subset(
        dset=CMIP5_TAS_FILE, space=(-10.0, 49.0, 10.0, 65.0), output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_time_and_space(tmpdir):
    """ Tests clisops subset function with time, space, level subsets."""
    result = subset(
        dset=CMIP5_TAS_FILE,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        space=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_tas(tmpdir):
    """ Tests with multiple tas files"""
    result = subset(
        dset=CMIP5_TAS,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        space=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_zostoga(tmpdir):
    """ Tests with multiple tas files"""
    result = subset(
        dset=CMIP5_ZOSTOGA,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_multiple_files_rh(tmpdir):
    """ Tests with multiple rh files"""
    result = subset(
        dset=CMIP5_RH,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


def test_subset_with_tas_series(tmpdir, tas_series):
    """ Test with tas_series fixture"""
    result = subset(
        dset=tas_series(["20", "22", "25"]),
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result
