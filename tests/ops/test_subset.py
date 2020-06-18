import os

import pytest

from .._common import CMIP5_ARCHIVE_BASE
from clisops.ops.subset import subset

TAS_NC = os.path.join(
    CMIP5_ARCHIVE_BASE,
    "cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc",  # noqa
)


def test_general_subset_dset(tmpdir):
    """ Tests clisops subset function with only a dataset"""
    result = subset(dset=TAS_NC, output_dir=tmpdir,)
    assert "output.nc" in result


def test_general_subset_time(tmpdir):
    """ Tests clisops subset function with a time subset."""
    result = subset(
        dset=TAS_NC,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


@pytest.mark.xfail(reason="test missing")
def test_general_subset_invalid_time():
    """ Tests clisops subset function with an invalid time subset."""
    assert False


def test_general_subset_space(tmpdir):
    """ Tests clisops subset function with a space subset."""
    result = subset(dset=TAS_NC, space=(0.0, 49.0, 10.0, 65.0), output_dir=tmpdir,)
    assert "output.nc" in result


@pytest.mark.xfail(reason="test missing")
def test_general_subset_invalid_space():
    """ Tests clisops subset function with an invalid space subset."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_general_subset_level():
    """ Tests clisops subset function with a level subset."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_general_subset_invalid_level():
    """ Tests clisops subset function with an invalid level subset."""
    assert False


def test_general_subset_all(tmpdir):
    """ Tests clisops subset function with time, space, level subsets."""
    result = subset(
        dset=TAS_NC,
        time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
        space=(0.0, 49.0, 10.0, 65.0),
        output_dir=tmpdir,
    )
    assert "output.nc" in result


@pytest.mark.xfail(reason="test missing")
def test_general_subset_file_type():
    """ Tests clisops api.general_subset function with a file type that isn't netcdf."""
    assert False
