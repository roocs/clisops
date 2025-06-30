import pytest

from clisops.exceptions import InvalidParameterValue
from clisops.parameter import AreaParameter, area

type_error = (
    "Input type of <{}> not allowed. Must be one of: "
    "[<class 'collections.abc.Sequence'>, <class 'str'>, <class 'clisops.parameter._utils.Series'>, <class 'NoneType'>]"
)


def test__str__():
    _area = "0.,49.,10.,65"
    parameter = AreaParameter(_area)
    assert parameter.__str__() == "Area to subset over:\n (0.0, 49.0, 10.0, 65.0)"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    _area = area("0.,49.,10.,65")
    parameter = AreaParameter(_area)
    assert parameter.__str__() == "Area to subset over:\n (0.0, 49.0, 10.0, 65.0)"
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()


def test_raw():
    _area = area("0.,49.,10.,65")
    parameter = AreaParameter(_area)
    assert parameter.raw == _area


def test_tuple():
    _area = "0.,49.,10.,65"
    parameter = AreaParameter(_area)
    assert parameter.value == (0.0, 49.0, 10.0, 65)


def test_area_is_tuple_string():
    _area = ("0", "-10", "120", "40")
    parameter = AreaParameter(_area)
    assert parameter.value == (0.0, -10.0, 120.0, 40.0)


def test_input_list():
    _area = [0, 49.5, 10, 65]
    parameter = AreaParameter(_area)
    assert parameter.value == (0.0, 49.5, 10.0, 65)


def test_validate_error_number():
    _area = 0
    with pytest.raises(InvalidParameterValue) as exc:
        AreaParameter(_area)
    assert str(exc.value) == type_error.format("class 'int'")


def test_validate_error_words():
    _area = ["test", "_area", "error", "words"]
    with pytest.raises(InvalidParameterValue) as exc:
        AreaParameter(_area)
    assert str(exc.value) == "Values must be valid numbers"


def test_validate_error_len_1_tuple():
    _area = (0, 65)
    with pytest.raises(InvalidParameterValue) as exc:
        AreaParameter(_area)
    assert str(exc.value) == "AreaParameter should be of length 4 but is of length 2"


def test_asdict():
    _area = "0.,49.,10.,65"
    parameter = AreaParameter(_area)
    assert parameter.asdict() == {"lon_bnds": (0, 10), "lat_bnds": (49, 65)}


def test_whitespace():
    _area = "0., 49., 10., 65"
    parameter = AreaParameter(_area)
    assert parameter.value == (0.0, 49.0, 10.0, 65)


def test_empty_string():
    _area = ""
    assert AreaParameter(_area).asdict() is None
    assert AreaParameter(_area).value is None


def test_none():
    _area = None
    assert AreaParameter(_area).asdict() is None
    assert AreaParameter(_area).value is None


def test_none_2():
    _area = None
    assert AreaParameter(_area).asdict() is None
    assert AreaParameter(_area).value is None


def test_class_instance():
    _area = "0.,49.,10.,65"
    parameter = AreaParameter(_area)
    new_parameter = AreaParameter(parameter)
    assert new_parameter.value == (0.0, 49.0, 10.0, 65.0)
