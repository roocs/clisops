import os

import pytest
import xarray as xr

from ._common import CMIP5_ARCHIVE_BASE
from clisops import utils


CMIP5_FPATHS = [
    os.path.join(
        CMIP5_ARCHIVE_BASE,
        "cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/*.nc",
    ),
    os.path.join(
        CMIP5_ARCHIVE_BASE,
        "cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
    ),
    os.path.join(
        CMIP5_ARCHIVE_BASE,
        "cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
    ),
]


def test_get_main_var_1():
    ds = xr.open_mfdataset(CMIP5_FPATHS[0])
    var_id = utils.get_coords.get_main_variable(ds)
    assert var_id == "zostoga"


def test_get_main_var_2():
    ds = xr.open_mfdataset(CMIP5_FPATHS[1])
    var_id = utils.get_coords.get_main_variable(ds)
    assert var_id == "tas"


def test_get_main_var_3():
    ds = xr.open_mfdataset(CMIP5_FPATHS[2])
    var_id = utils.get_coords.get_main_variable(ds)
    assert var_id == "rh"


@pytest.mark.xfail(reason="test missing")
def test_get_coord_by_attr_valid():
    """ Tests clisops utils.get_coord_by_attr with a real attribute e.g.
        standard_name or long_name"""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_coord_by_attr_invalid():
    """ Tests clisops utils.get_coord_by_attr with an attribute that
        doesn't exist."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_latitude():
    """ Tests clisops utils.get_latitude with a dataset that has
        a latitude coord with standard name latitude."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_latitude_fail():
    """ Tests clisops utils.get_latitude with a dataset on a coord that
    doesn't have the standard name latitude"""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_longitude():
    """ Tests clisops utils.get_longitude with a dataset that has
        a latitude coord with standard name longitude."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_longitude_fail():
    """ Tests clisops utils.get_longitude with a dataset on a coord that
    doesn't have the standard name longitude"""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_xy_no_space():
    """ Tests clisops utils._get_xy with a dataset but no space
        argument."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_xy_space():
    """ Tests clisops utils._get_xy with a dataset and space
        argument."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_get_xy_invalid_space():
    """ Tests clisops utils._get_xy with a dataset and space
        argument that is out of the range of the latitudes
        and longitudes."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_no_kwargs():
    """ Tests clisops.map_args with no kwargs. """
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_space():
    """ Tests clisops.map_args with only space kwarg."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_level():
    """ Tests clisops.map_args with only level kwarg."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_level_and_space():
    """ Tests clisops.map_args with level and space kwargs."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_include_time():
    """ Tests clisops.map_args with level and space and time kwargs."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_all_none():
    """ Tests clisops.map_args with level and space and time kwargs all set to None."""
    assert False


@pytest.mark.xfail(reason="test missing")
def test_map_args_invalid():
    """ Tests clisops.map_args with a kwarg that isn't level, space or time."""
    assert False
