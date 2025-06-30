import pytest

from clisops.exceptions import InvalidParameterValue
from clisops.parameter import DimensionParameter, dimensions


def test__str__():
    dims = "time,latitude"
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time', 'latitude')"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    dims = "time"
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time',)"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    # Make sequence from args
    dims = dimensions("time", "latitude")
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time', 'latitude')"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    # Make sequence from single arg
    dims = dimensions("time")
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time',)"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    # Make sequence from comma-separated string
    dims = dimensions("time,latitude")
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time', 'latitude')"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    # Make sequence from tuple of values
    dims = dimensions(("time", "latitude"))
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time', 'latitude')"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    # Make sequence from list of values
    dims = dimensions(["time", "latitude"])
    parameter = DimensionParameter(dims)
    assert parameter.__str__() == "Dimensions to average over:\n ('time', 'latitude')"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()


def test_raw():
    dims = "time,latitude"
    parameter = DimensionParameter(dims)
    assert parameter.raw == dims


def test_str():
    dims = "time,latitude"
    parameter = DimensionParameter(dims)
    assert parameter.value == ("time", "latitude")


def test_value():
    dims = ("time", "latitude")
    parameter = DimensionParameter(dims)
    assert parameter.value == ("time", "latitude")


def test_input_list():
    dims = ["time", "latitude"]
    parameter = DimensionParameter(dims)
    assert parameter.value == ("time", "latitude")


def test_validate_error_dimension():
    dims = "wrong"
    with pytest.raises(InvalidParameterValue) as exc:
        DimensionParameter(dims)
    assert (
        str(exc.value)
        == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude', 'realization']"
    )


def test_asdict():
    dims = ["time", "latitude"]
    parameter = DimensionParameter(dims)
    assert parameter.asdict() == {"dims": ("time", "latitude")}


def test_whitespace():
    dims = "time, latitude"
    parameter = DimensionParameter(dims)
    assert parameter.value == ("time", "latitude")


def test_empty_string():
    dims = ""
    assert DimensionParameter(dims).asdict() is None
    assert DimensionParameter(dims).value is None


def test_none():
    dims = None
    assert DimensionParameter(dims).asdict() is None
    assert DimensionParameter(dims).value is None


def test_class_instance():
    dims = "time"
    parameter = DimensionParameter(dims)
    new_parameter = DimensionParameter(parameter)
    assert new_parameter.value == ("time",)


def test_not_a_string():
    dims = (0, "latitude")
    with pytest.raises(InvalidParameterValue) as exc:
        DimensionParameter(dims)
    assert str(exc.value) == "Each dimension must be a string."
