import datetime

import pytest

from clisops.exceptions import InvalidParameterValue
from clisops.parameter import LevelParameter, level_interval, level_series

type_err = (
    "Input type of <{}> not allowed. Must be one of: "
    "[<class 'clisops.parameter.Interval'>, <class 'clisops.parameter.Series'>, <class 'NoneType'>]"
)


def test_interval_string_input():
    # start/
    parameter = LevelParameter("1000/")
    assert parameter.value == (1000, None)
    # /end
    parameter = LevelParameter("/2000")
    assert parameter.value == (None, 2000)
    # start/end
    parameter = LevelParameter("1000/2000")
    assert parameter.value == (1000, 2000)


def test_series_string_input():
    # one value
    parameter = LevelParameter("1000")
    assert parameter.value == [1000.0]
    # two values separated by ","
    parameter = LevelParameter("1000, 2000")
    assert parameter.value == [1000.0, 2000.0]
    # more than two values ...
    parameter = LevelParameter("1000, 2000, 3000")
    assert parameter.value == [1000.0, 2000.0, 3000.0]


def test__str__():
    level = level_interval("1000/2000")
    parameter = LevelParameter(level)
    assert (
        parameter.__str__()
        == "Level range to subset over\n first_level: 1000.0\n last_level: 2000.0"
    )
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()


def test_raw():
    level = level_interval("1000/2000")
    parameter = LevelParameter(level)
    assert parameter.raw == level


def test_validate_error_format():
    level = 1000
    with pytest.raises(InvalidParameterValue) as exc:
        LevelParameter(level)
    assert "not allowed" in str(exc.value)


def test_validate_error_len_1_tuple():
    with pytest.raises(InvalidParameterValue) as exc:
        level_interval((1000,))
    assert str(exc.value) == "Interval should be a range. Expected 2 values, received 1"


def test_not_numbers():
    level = level_interval(
        datetime.datetime(2085, 1, 1), datetime.datetime(2120, 12, 30)
    )
    with pytest.raises(InvalidParameterValue) as exc:
        LevelParameter(level)
    assert str(exc.value) == "Values must be valid numbers"


def test_word_string():
    level = level_interval("level/range")
    with pytest.raises(InvalidParameterValue) as exc:
        LevelParameter(level)
    assert str(exc.value) == "Values must be valid numbers"


def test_validate_error_no_slash():
    with pytest.raises(InvalidParameterValue) as exc:
        level_interval("1000 2000")
    assert str(exc.value) == "Interval should be passed in as a range separated by /"


def test_start_slash_end():
    level = level_interval("1000/2000")
    parameter = LevelParameter(level)
    assert parameter.value == (1000, 2000)


def test_float_string():
    level = level_interval("1000.50/2000.60")
    parameter = LevelParameter(level)
    assert parameter.value == (1000.5, 2000.6)


def test_float_tuple():
    level = level_interval(1000.50, 2000.60)
    parameter = LevelParameter(level)
    assert parameter.value == (1000.5, 2000.6)


def test_string_tuple():
    level = level_interval("1000.50", "2000.60")
    parameter = LevelParameter(level)
    assert parameter.value == (1000.5, 2000.6)


def test_int_tuple():
    level = level_interval(1000, 2000)
    parameter = LevelParameter(level)
    assert parameter.value == (1000, 2000)


def test_starting_slash():
    level = level_interval("1000/")
    parameter = LevelParameter(level)
    assert parameter.value == (1000, None)


def test_trailing_slash():
    level = level_interval("/2000")
    parameter = LevelParameter(level)
    assert parameter.value == (None, 2000)


def test_as_dict():
    level = level_interval("1000/2000")
    parameter = LevelParameter(level)
    assert parameter.asdict() == {"first_level": 1000, "last_level": 2000}


def test_slash_none():
    level = level_interval("/")
    parameter = LevelParameter(level)
    assert parameter.value is None
    assert parameter.asdict() == {"first_level": None, "last_level": None}


def test_none():
    level = None
    parameter = LevelParameter(level)
    assert parameter.value is None


def test_empty_string():
    level = level_interval("")
    parameter = LevelParameter(level)
    assert parameter.value is None

    with pytest.raises(InvalidParameterValue) as exc:
        LevelParameter("")
    assert "Unable to parse the level values entered" in str(exc.value)


def test_white_space():
    level = level_interval(" 1000 /2000")
    parameter = LevelParameter(level)
    assert parameter.value == (1000, 2000)


def test_class_instance():
    level = level_interval("1000/2000")
    parameter = LevelParameter(level)
    new_parameter = LevelParameter(parameter)
    assert new_parameter.value == (1000, 2000)


def test_level_series_input():
    value = [1000, 2000, 3000]
    vstring = ",".join([str(i) for i in value])

    for lev in (vstring, value, tuple(value)):
        level = level_series(lev)
        parameter = LevelParameter(level)
        assert parameter.type == "series"
        assert parameter.value == value
        assert parameter.asdict() == {"level_values": value}
