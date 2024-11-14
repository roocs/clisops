import pytest

from clisops.exceptions import InvalidParameterValue, MissingParameterValue
from clisops.parameter import CollectionParameter, collection

type_error = (
    "Input type of <{}> not allowed. Must be one of: "
    "[<class 'collections.abc.Sequence'>, <class 'str'>, <class 'clisops.parameter._utils.Series'>, <class 'clisops.utils.file_utils.FileMapper'>]"
)


def test__str__():
    coll = [
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    ]

    expected_str = (
        "Datasets to analyse:"
        "\ncmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
        "\ncmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
    )

    parameter = CollectionParameter(coll)
    assert parameter.__str__() == expected_str
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()

    coll = collection(
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    )

    parameter = CollectionParameter(coll)
    assert parameter.__str__() == expected_str
    assert parameter.__repr__() == parameter.__str__()
    assert parameter.__unicode__() == parameter.__str__()


def test_raw():
    coll = [
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    ]
    parameter = CollectionParameter(coll)
    assert parameter.raw == coll


def test_validate_error_id():
    coll = [
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        2,
    ]

    with pytest.raises(InvalidParameterValue) as exc:
        CollectionParameter(coll)
    assert (
        str(exc.value) == "Each id in a collection must be a string or "
        "an instance of <class 'clisops.utils.file_utils.FileMapper'>"
    )


def test_string():
    coll = (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga,"
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
    )

    parameter = CollectionParameter(coll)
    assert parameter.value == (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    )


def test_one_id():
    coll = "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
    parameter = CollectionParameter(coll)
    assert parameter.value == (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    )


def test_whitespace():
    coll = (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga, "
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga "
    )

    parameter = CollectionParameter(coll)
    assert parameter.value == (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    )


def test_empty_string():
    coll = ""
    with pytest.raises(MissingParameterValue) as exc:
        CollectionParameter(coll)
    assert str(exc.value) == "CollectionParameter must be provided"


def test_none():
    coll = None

    with pytest.raises(InvalidParameterValue) as exc:
        CollectionParameter(coll)
    assert str(exc.value) == type_error.format("class 'NoneType'")


def test_class_instance():
    coll = (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga,"
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
    )
    parameter = CollectionParameter(coll)
    new_parameter = CollectionParameter(parameter)
    assert new_parameter.value == (
        "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
        "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga",
    )
