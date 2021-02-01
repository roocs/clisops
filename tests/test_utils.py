import os

import pytest
from roocs_utils.exceptions import InvalidParameterValue


from clisops import CONFIG, utils


def test_local_config_loads():
    assert "clisops:read" in CONFIG
    assert "file_size_limit" in CONFIG["clisops:write"]


def test_dask_env_variables():
    assert os.getenv("MKL_NUM_THREADS") == "1"
    assert os.getenv("OPENBLAS_NUM_THREADS") == "1"
    assert os.getenv("OMP_NUM_THREADS") == "1"


def test_map_params(cmip5_tas_file):
    args = utils.map_params(
        ds=cmip5_tas_file,
        time=("1999-01-01T00:00:00", "2100-12-30T00:00:00"),
        area=(-5.0, 49.0, 10.0, 65),
        level=(1000.0, 1000.0),
    )

    # have a look at what date was used in clisops master
    assert args["start_date"] == "1999-01-01T00:00:00"
    assert args["end_date"] == "2100-12-30T00:00:00"
    assert args["lon_bnds"] == (-5, 10)
    assert args["lat_bnds"] == (49, 65)


def test_map_params_time(cmip5_tas_file):
    args = utils.map_params(
        ds=cmip5_tas_file, time=("1999-01-01", "2100-12"), area=(0, -90, 360, 90)
    )
    assert args["start_date"] == "1999-01-01T00:00:00"
    assert args["end_date"] == "2100-12-30T00:00:00"


def test_map_params_invalid_time(cmip5_tas_file):
    with pytest.raises(InvalidParameterValue):
        utils.map_params(
            ds=cmip5_tas_file,
            time=("1999-01-01T00:00:00", "maybe tomorrow"),
            area=(0, -90, 360, 90),
        )
    with pytest.raises(InvalidParameterValue):
        utils.map_params(ds=cmip5_tas_file, time=("", "2100"), area=(0, -90, 360, 90))


def test_map_params_area(cmip5_tas_file):
    args = utils.map_params(
        ds=cmip5_tas_file,
        area=(0, 10, 50, 60),
    )
    assert args["lon_bnds"] == (0, 50)
    assert args["lat_bnds"] == (10, 60)
    # allow also strings
    args = utils.map_params(
        ds=cmip5_tas_file,
        area=("0", "10", "50", "60"),
    )
    assert args["lon_bnds"] == (0, 50)
    assert args["lat_bnds"] == (10, 60)


def test_map_params_invalid_area(cmip5_tas_file):
    with pytest.raises(InvalidParameterValue):
        utils.map_params(
            ds=cmip5_tas_file,
            area=(0, 10, 50),
        )
    with pytest.raises(InvalidParameterValue):
        utils.map_params(
            ds=cmip5_tas_file,
            area=("zero", 10, 50, 60),
        )
