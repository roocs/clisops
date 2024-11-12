import datetime

import pytest

from clisops.exceptions import InvalidParameterValue
from clisops.parameter import time_interval, time_series
from clisops.parameter.time_parameter import TimeParameter

type_err = (
    "Input type of <{}> not allowed. Must be one of: "
    "[<class 'clisops.parameter.Interval'>, <class 'clisops.parameter.Series'>, <class 'NoneType'>]"
)


def test_interval_string_input():
    # start/
    parameter = TimeParameter("2085-01-01T12:00:00Z/")
    assert parameter.value == ("2085-01-01T12:00:00", None)
    assert parameter.get_bounds() == ("2085-01-01T12:00:00", None)
    # /end
    parameter = TimeParameter("/2120-12-30T12:00:00Z")
    assert parameter.value == (None, "2120-12-30T12:00:00")
    assert parameter.get_bounds() == (None, "2120-12-30T12:00:00")
    # start/end
    parameter = TimeParameter("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    assert parameter.value == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")
    assert parameter.get_bounds() == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")
    # start/end with non 360-day calendar
    parameter = TimeParameter("2085-01-01T00:00:00Z/2120-12-31T23:59:59Z")
    assert parameter.value == ("2085-01-01T00:00:00", "2120-12-31T23:59:59")
    assert parameter.get_bounds() == ("2085-01-01T00:00:00", "2120-12-31T23:59:59")
    # start/end with year
    parameter = TimeParameter("2085/2120")
    assert parameter.value == ("2085-01-01T00:00:00", "2120-12-31T23:59:59")
    assert parameter.get_bounds() == ("2085-01-01T00:00:00", "2120-12-31T23:59:59")
    # start/end with year-month-day
    parameter = TimeParameter("2085-01-16/2120-12-16")
    assert parameter.value == ("2085-01-16T00:00:00", "2120-12-16T23:59:59")
    assert parameter.get_bounds() == ("2085-01-16T00:00:00", "2120-12-16T23:59:59")


def test_series_string_input():
    # one value
    parameter = TimeParameter("2085-01-01T12:00:00Z")
    assert parameter.value == ["2085-01-01T12:00:00"]
    assert parameter.get_bounds() == ("2085-01-01T12:00:00", "2085-01-01T12:00:00")
    # two values separated by ","
    parameter = TimeParameter("2085-01-01T12:00:00Z, 2120-12-30T12:00:00Z")
    assert parameter.value == ["2085-01-01T12:00:00", "2120-12-30T12:00:00"]
    assert parameter.get_bounds() == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")
    # more then two values ...
    parameter = TimeParameter(
        "2085-01-01T12:00:00Z, 2090-01-01T12:00:00Z, 2120-12-30T12:00:00Z"
    )
    assert parameter.value == [
        "2085-01-01T12:00:00",
        "2090-01-01T12:00:00",
        "2120-12-30T12:00:00",
    ]
    assert parameter.get_bounds() == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")
    # with year only
    parameter = TimeParameter("2085, 2120")
    assert parameter.value == ["2085-01-01T00:00:00", "2120-01-01T00:00:00"]
    assert parameter.get_bounds() == ("2085-01-01T00:00:00", "2120-01-01T00:00:00")


def test__str__():
    time = time_interval("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert (
        parameter.__str__() == "Time period to subset over"
        "\n start time: 2085-01-01T12:00:00"
        "\n end time: 2120-12-30T12:00:00"
    )
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()


def test_raw():
    time = time_interval("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert parameter.raw == time


def test_validate_error_len_1_tuple():
    with pytest.raises(InvalidParameterValue) as exc:
        time_interval(("2085-01-01T12:00:00Z",))
    assert str(exc.value) == "Interval should be a range. Expected 2 values, received 1"


# should datetime objects be allowed?
def test_validate_error_datetime():
    time = datetime.datetime(2085, 1, 1)
    with pytest.raises(InvalidParameterValue) as exc:
        TimeParameter(time)
    assert "not allowed" in str(exc.value)


def test_datetime_tuple():
    time = time_interval(datetime.datetime(2085, 1, 1), datetime.datetime(2120, 12, 30))
    with pytest.raises(InvalidParameterValue) as exc:
        TimeParameter(time)
    assert str(exc.value) == "Unable to parse the time values entered"


def test_validate_error_no_slash():
    with pytest.raises(InvalidParameterValue) as exc:
        time_interval("2085-01-01T12:00:00Z 2120-12-30T12:00:00Z")
    assert str(exc.value) == ("Interval should be passed in as a range separated by /")


def test_trailing_slash():
    time = time_interval("2085-01-01T12:00:00Z/")
    parameter = TimeParameter(time)
    assert parameter.value == ("2085-01-01T12:00:00", None)


def test_starting_slash():
    time = time_interval("/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert parameter.value == (None, "2120-12-30T12:00:00")


def test_start_slash_end():
    time = time_interval("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert parameter.value == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")


def test_as_dict():
    time = time_interval("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert parameter.asdict() == {
        "start_time": "2085-01-01T12:00:00",
        "end_time": "2120-12-30T12:00:00",
    }


def test_slash_none():
    time = time_interval("/")
    parameter = TimeParameter(time)
    assert parameter.value is None
    assert parameter.asdict() == {"start_time": None, "end_time": None}


def test_none():
    time = None
    parameter = TimeParameter(time)
    assert parameter.value is None


def test_empty_string():
    time = time_interval("")
    parameter = TimeParameter(time)
    assert parameter.value is None

    with pytest.raises(InvalidParameterValue) as exc:
        TimeParameter("")
    assert "Unable to parse the time values entered" in str(exc.value)


def test_white_space():
    time = time_interval("2085-01-01T12:00:00Z / 2120-12-30T12:00:00Z ")
    parameter = TimeParameter(time)
    assert parameter.value == ("2085-01-01T12:00:00", "2120-12-30T12:00:00")


def test_class_instance():
    time = time_interval("2085-01-01T12:00:00Z/2120-12-30T12:00:00Z")
    parameter = TimeParameter(time)
    new_parameter = TimeParameter(parameter)
    assert new_parameter.value == (
        "2085-01-01T12:00:00",
        "2120-12-30T12:00:00",
    )


def test_360_day_calendar():
    time = time_interval("2007-02-29T12:00:00Z/2010-02-30T12:00:00Z")
    parameter = TimeParameter(time)
    assert parameter.value == ("2007-02-29T12:00:00", "2010-02-30T12:00:00")

    time = time_series(
        "2007-02-29T12:00:00Z", "2009-02-30T12:00:00Z", "2010-02-30T12:00:00Z"
    )
    parameter = TimeParameter(time)
    assert parameter.value == [
        "2007-02-29T12:00:00",
        "2009-02-30T12:00:00",
        "2010-02-30T12:00:00",
    ]


def test_time_series_input():
    value = ["2085-01-01T12:00:00Z", "2095-03-03T03:03:03", "2120-12-30T12:00:00Z"]
    expected_value = [i.replace("Z", "") for i in value]
    vstring = ",".join([str(i) for i in value])

    for tm in (vstring, value, tuple(value)):
        times = time_series(tm)
        parameter = TimeParameter(times)
        assert parameter.type == "series"
        assert parameter.value == expected_value
        assert parameter.asdict() == {"time_values": expected_value}

    times = time_series(
        "2085-01-01T12:00:00Z", "2095-03-03T03:03:03", "2120-12-30T12:00:00Z"
    )
    parameter = TimeParameter(times)
    assert parameter.type == "series"
    assert parameter.value == expected_value
    assert parameter.asdict() == {"time_values": expected_value}


def test_time_parameter_get_bounds():
    # Tests that the get_bounds() method of TimeParameter is working for
    # types "series", "interval" and "none"
    t_values = "2085-01-01T12:00:00", "2095-03-03T03:03:03", "2120-12-30T12:00:00"
    t_bounds = t_values[0], t_values[-1]

    def are_equal(b0, b1):
        return b0[0][:19] == b1[0][:19] and b0[1][:19] == b1[1][:19]

    parameter = TimeParameter(time_series(t_values))
    assert parameter.type == "series"
    assert are_equal(parameter.get_bounds(), t_bounds)

    parameter = TimeParameter(time_series(t_bounds))
    assert parameter.type == "series"
    assert are_equal(parameter.get_bounds(), t_bounds)

    parameter = TimeParameter(None)
    assert parameter.type == "none"
    assert parameter.get_bounds() == (None, None)
