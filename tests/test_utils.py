import pytest
from dateutil.parser import ParserError

from clisops import utils
from clisops.exceptions import InvalidParameterValue, MissingParameterValue


def test_parse_date():
    assert "2020-05-19" == utils.parse_date("2020-05-19")
    assert "1999-01-01" == utils.parse_date("1999-01-01T00:00:00")
    with pytest.raises(ParserError):
        utils.parse_date("tomorrow")


def test_parse_date_year():
    assert "2020" == utils.parse_date_year("2020-05-20")
    with pytest.raises(ParserError):
        utils.parse_date_year("yesterday")


def test_map_params():
    args = utils.map_params(
        time=("1999-01-01T00:00:00", "2100-12-30T00:00:00"),
        space=(-5.0, 49.0, 10.0, 65),
        level=(1000.0,),
    )
    assert args["start_date"] == "1999"
    assert args["end_date"] == "2100"
    assert args["lon_bnds"] == (-5, 10)
    assert args["lat_bnds"] == (49, 65)


def test_map_params_time():
    args = utils.map_params(time=("1999-01-01", "2100-12"),)
    assert args["start_date"] == "1999"
    assert args["end_date"] == "2100"


def test_map_params_invalid_time():
    with pytest.raises(InvalidParameterValue):
        utils.map_params(time=("1999-01-01T00:00:00", "maybe tomorrow"),)
    with pytest.raises(InvalidParameterValue):
        utils.map_params(time=("", "2100"),)


def test_map_params_space():
    args = utils.map_params(space=(0, 10, 50, 60),)
    assert args["lon_bnds"] == (0, 50)
    assert args["lat_bnds"] == (10, 60)
    # allow also strings
    args = utils.map_params(space=("0", "10", "50", "60"),)
    assert args["lon_bnds"] == (0, 50)
    assert args["lat_bnds"] == (10, 60)


def test_map_params_invalid_space():
    with pytest.raises(InvalidParameterValue):
        utils.map_params(space=(0, 10, 50),)
    with pytest.raises(InvalidParameterValue):
        utils.map_params(space=("zero", 10, 50, 60),)


def test_map_params_missing_param():
    with pytest.raises(MissingParameterValue):
        utils.map_params()
